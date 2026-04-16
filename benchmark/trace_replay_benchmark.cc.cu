/*
 * Exp #5 — Trace replay benchmark (experiment.md §4 «Continuous Operation Under
 * Real Workload», §5.3 sketch, §7 constraints).
 *
 * Trace: one or more binary files concatenated in order; each file is a dense
 * sequence of uint64 little-endian feature IDs (experiment.md §7).
 *
 * Protocol (experiment.md §4):
 *   Phase 1 — Warm-up (first warmup_fraction of key *accesses*, default 20%):
 *     For each batch: find(batch) → insert_or_assign(batch)  [same shape as
 *     steady so we can record hit_rate; §4 lists hit rate for Phase 1.]
 *     Table starts empty; both cache and baseline modes fill naturally.
 *   Phase 2 — Steady (remaining trace):
 *     For each batch of batch_size keys (default 1M, §7):
 *       find(batch) → record per-batch hit rate
 *       insert_or_assign(batch) → included in wall-clock / throughput
 *     Cache (HKV): inline eviction only.
 *     Baseline: after the batch, if load_factor > evict_lf_trigger (default 0.9),
 *       run export_batch → CPU sort by ascending score → erase bottom
 *       evict_fraction of *exported* rows (experiment.md §4 pseudo-code).
 *
 * Timing: all wall-clock includes GPU sync, baseline export/sort/erase on CPU
 * (experiment.md §7).
 *
 * CSV (extends experiment.md §5.3 listing with phase + cumulative hit rate):
 *   phase,batch,wall_clock_ms,throughput_bkvs,hit_rate,cumulative_hit_rate,
 *   load_factor,stall_occurred,stall_duration_ms,cumulative_ops
 *
 * Defaults: dim=32, batch_size=1M (experiment.md §7 Config B).
 *
 * Optional: --max-bucket-size N (N>0) sets HashTableOptions::max_bucket_size;
 * use with --capacity N for a single-bucket-style layout (Exp #5 new_baseline).
 *
 * Example:
 *   ./trace_replay_benchmark --trace day_0.bin --capacity 10000000 \
 *     --mode cache --strategy lru --out results.csv
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;

using namespace nv::merlin;
using namespace benchmark;

/* ---------- Trace: mmap each file; logical concatenation ---------- */

struct TraceSegment {
  const K* keys = nullptr;
  size_t num_keys = 0;
  void* mapped = nullptr;
  size_t byte_len = 0;
  int fd = -1;

  bool map_file(const char* path) {
    fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st {};
    if (fstat(fd, &st) != 0) return false;
    byte_len = static_cast<size_t>(st.st_size);
    if (byte_len % sizeof(K) != 0) {
      std::cerr << "warning: " << path
                 << " size not multiple of 8; trailing bytes dropped\n";
      byte_len -= byte_len % sizeof(K);
    }
    if (byte_len == 0) return false;
    mapped = mmap(nullptr, byte_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
      mapped = nullptr;
      return false;
    }
    keys = reinterpret_cast<const K*>(mapped);
    num_keys = byte_len / sizeof(K);
    return true;
  }

  void unmap() {
    if (mapped && mapped != MAP_FAILED) {
      munmap(mapped, byte_len);
      mapped = nullptr;
      keys = nullptr;
      num_keys = 0;
      byte_len = 0;
    }
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  ~TraceSegment() { unmap(); }
};

struct TraceConcat {
  std::vector<TraceSegment> parts;
  size_t total_keys = 0;

  bool open(const std::vector<std::string>& paths) {
    total_keys = 0;
    parts.clear();
    parts.resize(paths.size());
    for (size_t i = 0; i < paths.size(); i++) {
      if (!parts[i].map_file(paths[i].c_str())) {
        std::cerr << "mmap failed: " << paths[i] << "\n";
        for (size_t j = 0; j < i; j++) parts[j].unmap();
        parts.clear();
        total_keys = 0;
        return false;
      }
      total_keys += parts[i].num_keys;
    }
    return total_keys > 0;
  }

  void copy_keys_to_host_buffer(K* dst, size_t global_off, size_t count) const {
    size_t written = 0;
    size_t pos = global_off;
    while (written < count) {
      size_t skip = 0;
      const TraceSegment* seg = nullptr;
      size_t seg_local_start = 0;
      for (const auto& p : parts) {
        if (pos < skip + p.num_keys) {
          seg = &p;
          seg_local_start = pos - skip;
          break;
        }
        skip += p.num_keys;
      }
      if (!seg) break;
      size_t n = std::min(count - written, seg->num_keys - seg_local_start);
      memcpy(dst + written, seg->keys + seg_local_start, n * sizeof(K));
      written += n;
      pos += n;
    }
  }
};

struct Args {
  std::vector<std::string> trace_paths;
  const char* out_csv = "trace_replay_results.csv";
  size_t capacity = 0;
  int dim = 32;
  size_t batch_size = 1024 * 1024;
  double warmup_fraction = 0.2;
  bool baseline = false;
  bool use_lfu = false;
  float evict_lf_trigger = 0.90f;
  float evict_fraction = 0.25f;
  size_t hbm_gb = 128;
  size_t max_keys = 0;
  /** 0 = Merlin default (128); >0 must be power-of-two and satisfy Merlin bucket layout checks. */
  size_t max_bucket_size = 0;
};

static void print_usage() {
  std::cerr
      << "trace_replay_benchmark [options]\n"
      << "  (experiment.md §4 Exp #5, §5.3, §7: dim=32, batch 1M defaults)\n"
      << "  --trace PATH          uint64 LE trace (repeat for multiple files, "
         "concatenated in order)\n"
      << "  --out PATH            output CSV\n"
      << "  --capacity N          init_capacity = max_capacity = N (required)\n"
      << "  --dim N               embedding dim (default: 32)\n"
      << "  --batch-size N        keys per batch (default: 1048576)\n"
      << "  --warmup-fraction F   first F of key accesses = Phase 1 (default: 0.2)\n"
      << "  --mode MODE           cache | baseline (default: cache)\n"
      << "  --strategy STR        lru | lfu (default: lru)\n"
      << "  --evict-lf-trigger F  baseline: eviction if LF > F after batch (default: "
         "0.9)\n"
      << "  --evict-fraction F    baseline: erase floor(N_exported * F) lowest "
         "scores (default: 0.25)\n"
      << "  --hbm-gb N            max_hbm_for_vectors (default: 128)\n"
      << "  --max-keys N          cap total key accesses (0 = all; debug)\n"
      << "  --max-bucket-size N   Merlin max_bucket_size (0 = default 128)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (k == "--trace") {
      const char* v = need("--trace");
      if (!v) return false;
      a.trace_paths.push_back(v);
    } else if (k == "--out")
      a.out_csv = need("--out");
    else if (k == "--capacity")
      a.capacity = static_cast<size_t>(strtoull(need("--capacity"), nullptr, 10));
    else if (k == "--dim")
      a.dim = atoi(need("--dim"));
    else if (k == "--batch-size")
      a.batch_size = static_cast<size_t>(strtoull(need("--batch-size"), nullptr, 10));
    else if (k == "--warmup-fraction")
      a.warmup_fraction = strtod(need("--warmup-fraction"), nullptr);
    else if (k == "--mode") {
      const char* v = need("--mode");
      a.baseline = (std::string(v) == "baseline");
    } else if (k == "--strategy") {
      const char* v = need("--strategy");
      a.use_lfu = (std::string(v) == "lfu");
    } else if (k == "--evict-lf-trigger")
      a.evict_lf_trigger = static_cast<float>(strtod(need("--evict-lf-trigger"), nullptr));
    else if (k == "--evict-fraction")
      a.evict_fraction = static_cast<float>(strtod(need("--evict-fraction"), nullptr));
    else if (k == "--hbm-gb")
      a.hbm_gb = static_cast<size_t>(strtoull(need("--hbm-gb"), nullptr, 10));
    else if (k == "--max-keys")
      a.max_keys = static_cast<size_t>(strtoull(need("--max-keys"), nullptr, 10));
    else if (k == "--max-bucket-size")
      a.max_bucket_size = static_cast<size_t>(strtoull(need("--max-bucket-size"), nullptr, 10));
    else if (k == "-h" || k == "--help") {
      print_usage();
      return false;
    } else {
      std::cerr << "unknown arg: " << k << "\n";
      return false;
    }
  }
  if (a.trace_paths.empty() || a.capacity == 0) {
    std::cerr << "at least one --trace and --capacity are required.\n";
    print_usage();
    return false;
  }
  if (a.max_bucket_size > 0) {
    auto is_pow2 = [](size_t x) -> bool { return x > 0 && (x & (x - 1)) == 0; };
    if (!is_pow2(a.max_bucket_size)) {
      std::cerr << "--max-bucket-size must be a power of two (got " << a.max_bucket_size << ")\n";
      return false;
    }
    if (a.max_bucket_size < 128) {
      std::cerr << "--max-bucket-size must be >= 128\n";
      return false;
    }
    if (a.capacity < a.max_bucket_size) {
      std::cerr << "--capacity must be >= --max-bucket-size for this layout\n";
      return false;
    }
  }
  return true;
}

/* experiment.md §4 baseline: export → sort indices by score ascending → erase */

template <int Strategy>
static int run_typed(const Args& args, TraceConcat& trace) {
  using Table = HashTable<K, V, S, Strategy, Sm80>;

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));

  HashTableOptions options;
  options.init_capacity = args.capacity;
  options.max_capacity = args.capacity;
  options.dim = static_cast<size_t>(args.dim);
  options.max_load_factor = 1.0f;
  options.max_hbm_for_vectors = nv::merlin::GB(args.hbm_gb);
  if (args.max_bucket_size > 0) {
    options.max_bucket_size = args.max_bucket_size;
  }

  auto table = std::make_shared<Table>();
  table->init(options);
  std::cerr << "trace_replay_benchmark: capacity=" << args.capacity
            << " max_bucket_size=" << (args.max_bucket_size > 0 ? args.max_bucket_size : size_t{128})
            << " mode=" << (args.baseline ? "baseline" : "cache") << "\n";

  const size_t batch = args.batch_size;
  const size_t nkeys =
      (args.max_keys > 0) ? std::min(args.max_keys, trace.total_keys) : trace.total_keys;
  /* Phase 1 = first warmup_fraction of key accesses (experiment.md §4). */
  const size_t n_warmup = static_cast<size_t>(
      std::floor(static_cast<double>(nkeys) * std::min(1.0, std::max(0.0, args.warmup_fraction))));

  K* d_keys = nullptr;
  V* d_vectors = nullptr;
  S* d_scores = nullptr;
  bool* d_found = nullptr;
  bool* h_found = nullptr;
  K* h_keys_batch = nullptr;

  CUDA_CHECK(cudaMallocHost(&h_found, batch * sizeof(bool)));
  CUDA_CHECK(cudaMallocHost(&h_keys_batch, batch * sizeof(K)));

  CUDA_CHECK(cudaMalloc(&d_keys, batch * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, batch * static_cast<size_t>(args.dim) * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_found, batch * sizeof(bool)));

  const bool need_scores = (Strategy == EvictStrategy::kLfu) ||
                           (Strategy == EvictStrategy::kEpochLfu) ||
                           (Strategy == EvictStrategy::kCustomized);
  if (need_scores) {
    CUDA_CHECK(cudaMalloc(&d_scores, batch * sizeof(S)));
  } else {
    d_scores = nullptr;
  }

  K* d_export_keys = nullptr;
  V* d_export_vals = nullptr;
  S* d_export_scores = nullptr;
  K* d_evict_keys = nullptr;
  if (args.baseline) {
    CUDA_CHECK(cudaMalloc(&d_export_keys, args.capacity * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_export_vals,
                          args.capacity * static_cast<size_t>(args.dim) * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_export_scores, args.capacity * sizeof(S)));
    CUDA_CHECK(cudaMalloc(&d_evict_keys, args.capacity * sizeof(K)));
  }

  std::vector<K> h_export_keys;
  std::vector<S> h_export_scores;
  std::vector<size_t> order;

  std::ofstream csv(args.out_csv);
  if (!csv) {
    std::cerr << "cannot open output: " << args.out_csv << "\n";
    return 1;
  }
  csv << "phase,batch,wall_clock_ms,throughput_bkvs,hit_rate,cumulative_hit_rate,"
         "load_factor,stall_occurred,stall_duration_ms,cumulative_ops\n";

  auto baseline_eviction_cycle = [&](double* stall_ms_out) {
    *stall_ms_out = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();

    const size_t dumped = table->export_batch(args.capacity, 0, d_export_keys, d_export_vals,
                                             d_export_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (dumped == 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      *stall_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
      return;
    }

    h_export_keys.resize(dumped);
    h_export_scores.resize(dumped);
    CUDA_CHECK(cudaMemcpy(h_export_keys.data(), d_export_keys, dumped * sizeof(K),
                          cudaMemcpyDeviceToHost));
    if (d_export_scores) {
      CUDA_CHECK(cudaMemcpy(h_export_scores.data(), d_export_scores, dumped * sizeof(S),
                            cudaMemcpyDeviceToHost));
    } else {
      std::fill(h_export_scores.begin(), h_export_scores.end(), S{0});
    }

    order.resize(dumped);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t aa, size_t bb) {
      return h_export_scores[aa] < h_export_scores[bb];
    });

    /* n_evict = floor(dumped * evict_fraction), experiment.md §4 */
    size_t n_evict = static_cast<size_t>(static_cast<double>(dumped) *
                                         static_cast<double>(args.evict_fraction));
    if (n_evict == 0 && dumped > 0) n_evict = 1;
    n_evict = std::min(n_evict, dumped);

    std::vector<K> h_evict(n_evict);
    for (size_t i = 0; i < n_evict; i++) {
      h_evict[i] = h_export_keys[order[i]];
    }
    CUDA_CHECK(cudaMemcpy(d_evict_keys, h_evict.data(), n_evict * sizeof(K),
                          cudaMemcpyHostToDevice));
    table->erase(n_evict, d_evict_keys, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    *stall_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  auto run_batch = [&](size_t cur, size_t global_off, bool is_steady, size_t batch_idx,
                       uint64_t& cumulative, uint64_t& steady_hits, uint64_t& steady_seen,
                       double& cum_hit_rate_steady) {
    trace.copy_keys_to_host_buffer(h_keys_batch, global_off, cur);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys_batch, cur * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vectors, 0, cur * static_cast<size_t>(args.dim) * sizeof(V)));
    if (need_scores) {
      std::vector<S> hsc(cur, 1);
      CUDA_CHECK(cudaMemcpy(d_scores, hsc.data(), cur * sizeof(S), cudaMemcpyHostToDevice));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    table->set_global_epoch(static_cast<int>(batch_idx));

    table->find(cur, d_keys, d_vectors, d_found, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(
        cudaMemcpy(h_found, d_found, cur * sizeof(bool), cudaMemcpyDeviceToHost));
    size_t hits = 0;
    for (size_t i = 0; i < cur; i++) {
      if (h_found[i]) hits++;
    }

    table->set_global_epoch(static_cast<int>(batch_idx));
    table->insert_or_assign(cur, d_keys, d_vectors, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cumulative += cur;

    float lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool stalled = false;
    double stall_ms = 0.0;
    if (args.baseline && lf > args.evict_lf_trigger) {
      baseline_eviction_cycle(&stall_ms);
      stalled = true;
      ms += stall_ms;
      lf = table->load_factor(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const double tp =
        (ms > 0.0) ? (2.0 * static_cast<double>(cur) / (ms * 1e-3) / 1e9) : 0.0;
    const float hit_rate = static_cast<float>(hits) / static_cast<float>(cur);

    if (is_steady) {
      steady_hits += hits;
      steady_seen += cur;
      cum_hit_rate_steady =
          steady_seen ? static_cast<double>(steady_hits) / static_cast<double>(steady_seen)
                      : 0.0;
    }

    const char* phase = is_steady ? "steady" : "warmup";
    const float cum_csv =
        is_steady ? static_cast<float>(cum_hit_rate_steady) : -1.f;

    csv << phase << "," << batch_idx << "," << std::setprecision(6) << ms << "," << tp << ","
        << hit_rate << ",";
    if (is_steady)
      csv << cum_csv;
    else
      csv << "-1";
    csv << "," << lf << "," << (stalled ? 1 : 0) << "," << stall_ms << "," << cumulative
        << "\n";
  };

  uint64_t cumulative = 0;
  uint64_t steady_hits = 0;
  uint64_t steady_seen = 0;
  double cum_hit_rate_steady = 0.0;

  size_t global_off = 0;
  size_t w_batch = 0;
  while (global_off < n_warmup) {
    const size_t cur = std::min(batch, n_warmup - global_off);
    run_batch(cur, global_off, false, w_batch, cumulative, steady_hits, steady_seen,
              cum_hit_rate_steady);
    global_off += cur;
    w_batch++;
  }

  size_t s_batch = 0;
  while (global_off < nkeys) {
    const size_t cur = std::min(batch, nkeys - global_off);
    run_batch(cur, global_off, true, s_batch, cumulative, steady_hits, steady_seen,
              cum_hit_rate_steady);
    global_off += cur;
    s_batch++;
  }

  if (d_export_keys) cudaFree(d_export_keys);
  if (d_export_vals) cudaFree(d_export_vals);
  if (d_export_scores) cudaFree(d_export_scores);
  if (d_evict_keys) cudaFree(d_evict_keys);
  if (d_scores) cudaFree(d_scores);
  cudaFree(d_keys);
  cudaFree(d_vectors);
  cudaFree(d_found);
  cudaFreeHost(h_found);
  cudaFreeHost(h_keys_batch);
  cudaStreamDestroy(stream);

  std::cout << "Wrote " << args.out_csv << " (warmup_batches=" << w_batch
            << " steady_batches=" << s_batch << ")\n";
  return 0;
}

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) return 1;

  TraceConcat trace;
  if (!trace.open(args.trace_paths)) {
    return 1;
  }

  if (args.use_lfu) {
    return run_typed<EvictStrategy::kLfu>(args, trace);
  }
  return run_typed<EvictStrategy::kLru>(args, trace);
}
