/*
 * E6: Digest Ablation — find throughput with/without digest filtering
 *
 * Configs A/B/C at λ=0.50 and λ=1.00
 * Compile normally for "with digest", with -DDISABLE_DIGEST for "without".
 * 5 runs per (config, lf) point.
 *
 * Output: CSV with mode, config, load_factor, run, throughput.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using namespace nv::merlin;
using namespace benchmark;

static constexpr size_t BATCH_SIZE = 1024 * 1024UL;
static constexpr int NUM_RUNS = 5;
static constexpr int NUM_WARMUP = 3;
static constexpr float EPSILON = 0.001f;

struct Config {
  const char* name;
  size_t dim;
  size_t capacity;
  size_t hbm_gb;
};

using HKVTable = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

float measure_find(size_t dim, size_t init_capacity, size_t hbm_gb,
                   float load_factor, cudaStream_t stream) {
  HashTableOptions options;
  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(hbm_gb);

  auto table = std::make_shared<HKVTable>();
  table->init(options);

  K* h_keys;
  S* h_scores;
  K* d_keys;
  V* d_vectors;
  bool* d_found;
  V** d_vptrs;

  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, BATCH_SIZE * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_vptrs, BATCH_SIZE * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * dim));

  // Pre-populate
  size_t target = static_cast<size_t>(init_capacity * load_factor);
  K start = 0;
  int epoch = 0;
  while (start < target) {
    size_t cur = std::min(BATCH_SIZE, target - start);
    table->set_global_epoch(epoch++);
    create_continuous_keys<K, S>(h_keys, h_scores, cur, start);
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    table->find_or_insert(cur, d_keys, d_vptrs, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  // Fine-tune LF
  float real_lf = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (load_factor - real_lf > EPSILON) {
    auto append =
        static_cast<int64_t>((load_factor - real_lf) * init_capacity);
    if (append <= 0) break;
    append = std::min(static_cast<int64_t>(BATCH_SIZE), append);
    create_continuous_keys<K, S>(h_keys, h_scores, append, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, append * sizeof(K),
                           cudaMemcpyHostToDevice));
    table->insert_or_assign(append, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += append;
    real_lf = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Prepare 100% hit keys for pure throughput measurement
  create_continuous_keys<K, S>(h_keys, h_scores, BATCH_SIZE, 0);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                         cudaMemcpyHostToDevice));

  // Warmup
  for (int w = 0; w < NUM_WARMUP; w++) {
    table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Timed run
  auto timer = benchmark::Timer<double>();
  timer.start();
  table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.end();

  float throughput =
      BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_vptrs));

  return throughput;
}

int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
#ifdef DISABLE_DIGEST
  const char* mode = "no_digest";
  std::cerr << "Mode: DIGEST DISABLED (ablation)" << std::endl;
#else
  const char* mode = "with_digest";
  std::cerr << "Mode: DIGEST ENABLED (default)" << std::endl;
#endif

  std::vector<Config> configs = {
      {"A", 8, 128UL * 1024 * 1024, 4},
      {"B", 32, 128UL * 1024 * 1024, 16},
      {"C", 64, 64UL * 1024 * 1024, 16},
  };
  std::vector<float> load_factors = {0.50f, 1.00f};

  std::cout << "mode,config,dim,load_factor,run,throughput_bkvs" << std::endl;

  try {
    for (auto& cfg : configs) {
      for (float lf : load_factors) {
        for (int run = 0; run < NUM_RUNS; run++) {
          std::cerr << "  " << mode << " " << cfg.name << " LF=" << std::fixed
                    << std::setprecision(2) << lf << " run=" << (run + 1)
                    << "/" << NUM_RUNS << "..." << std::flush;

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));

          float tp =
              measure_find(cfg.dim, cfg.capacity, cfg.hbm_gb, lf, stream);

          std::cout << mode << "," << cfg.name << "," << cfg.dim << ","
                    << std::fixed << std::setprecision(2) << lf << ","
                    << (run + 1) << "," << std::setprecision(6) << tp
                    << std::endl;

          std::cerr << " " << std::fixed << std::setprecision(3) << tp
                    << " B-KV/s" << std::endl;

          CUDA_CHECK(cudaStreamDestroy(stream));
        }
      }
    }
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
