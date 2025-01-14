/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

using namespace nv::merlin;
using namespace benchmark;

enum class Test_Mode {
  pure_hbm = 0,
  hybrid = 1,
};

const float EPSILON = 0.001f;

std::string rep(int n) { return std::string(n, ' '); }

using K = uint64_t;
using S = uint64_t;
using V = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;


template <class K, class S>
void create_uniform_keys(K* h_keys, S* h_scores, std::deque<K>& queue_keys, const int key_num_per_op, const K start = 0, int freq_range = 1024 * 1024) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < key_num_per_op) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    queue_keys.push_back(num);
    if (queue_keys.size() > freq_range) queue_keys.pop_front();
    if (h_scores != nullptr) h_scores[i] = (start + static_cast<K>(i)) % freq_range;
    i++;
  }
}

template <class Table>
float test_one_api(std::shared_ptr<Table>& table, const API_Select api,
                   const size_t dim, const size_t capacity,
                   const size_t batch_size, const float load_factor,
                   const float hitrate = 0.95f) {
  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, batch_size * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, batch_size * sizeof(S)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, batch_size * sizeof(V) * dim));
  CUDA_CHECK(cudaMallocHost(&h_found, batch_size * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, batch_size * sizeof(V) * dim));

  bool need_scores = (Table::evict_strategy == EvictStrategy::kLfu ||
                      Table::evict_strategy == EvictStrategy::kEpochLfu ||
                      Table::evict_strategy == EvictStrategy::kCustomized);

  K* d_keys;
  S* d_scores_real;
  S* d_scores;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  K* d_keys_out;

  K* d_evict_keys;
  S* d_evict_scores;

  CUDA_CHECK(cudaMalloc(&d_keys, batch_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_real, batch_size * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, batch_size * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, batch_size * sizeof(V) * dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, batch_size * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, batch_size * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, batch_size * sizeof(K)));

  CUDA_CHECK(cudaMalloc(&d_evict_keys, batch_size * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_scores, batch_size * sizeof(S)));

  CUDA_CHECK(cudaMemset(d_vectors, 1, batch_size * sizeof(V) * dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, batch_size * sizeof(V) * dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, batch_size * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, batch_size * sizeof(bool)));

  d_scores = need_scores ? d_scores_real : nullptr;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // std::unordered_set<K> exist_keys;
  std::deque<K> queue_keys;
  // std::random_device rd;
  // std::mt19937_64 eng(rd());
  // std::uniform_int_distribution<K> distr;
  int freq_range = batch_size;
  // size_t range = std::numeric_limits<uint64_t>::max();

  // assert(exist_keys.size() == 0 && stack_keys.size() == 0);
 
  // initialize insert
  // step 1, no need to load load_factor
  uint64_t key_num_init = static_cast<uint64_t>(capacity * load_factor);
  const float target_load_factor = key_num_init * 1.0f / capacity;
  uint64_t key_num_remain = key_num_init % batch_size == 0
                                ? batch_size
                                : key_num_init % batch_size;
  int32_t loop_num_init = (key_num_init + batch_size - 1) / batch_size;


  // K start = 0UL;
  S threshold = benchmark::host_nano<S>();
  int global_epoch = 0;
  for (; global_epoch < loop_num_init; global_epoch++) {
    table->set_global_epoch(global_epoch);
    uint64_t key_num_cur_insert =
        global_epoch == loop_num_init - 1 ? key_num_remain : batch_size;
    // create_continuous_keys<K, S>(h_keys, h_scores, key_num_cur_insert, start);
    create_uniform_keys<K,S>(h_keys, h_scores, queue_keys, key_num_cur_insert, 0, freq_range);
    // {
    //   uint64_t target_key_num = exist_keys.size() + key_num_cur_insert;
    //   int i = 0;
    //   while (exist_keys.size() < target_key_num) {
    //     K new_key = distr(eng) % range;
    //     if (exist_keys.find(new_key) == exist_keys.end()) {
    //       stack_keys.push(new_key);
    //       h_keys[i] = new_key;
    //       h_scores[i] = static_cast<K>(i) % freq_range;
    //       i++;
    //     }
    //     exist_keys.insert(new_key);
    //   }
    //   assert(i == key_num_cur_insert);
    // }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_cur_insert * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores_real, h_scores,
                          key_num_cur_insert * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->find_or_insert(key_num_cur_insert, d_keys, d_vectors_ptr, d_found,
                          d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // start += key_num_cur_insert;
  }

  // step 2
  table->set_global_epoch(global_epoch);
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  K start = 0UL;
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * capacity);
    if (key_num_append <= 0) break;
    key_num_append =
        std::min(static_cast<int64_t>(batch_size), key_num_append);
    // create_continuous_keys<K, S>(h_keys, h_scores, key_num_append, start);
    create_uniform_keys<K,S>(h_keys, h_scores, queue_keys, key_num_append, start, freq_range);
    // {
    //   uint64_t target_key_num = exist_keys.size() + key_num_append;
    //   int i = start;
    //   while (exist_keys.size() < target_key_num) {
    //     K new_key = distr(eng) % range;
    //     if (exist_keys.find(new_key) == exist_keys.end()) {
    //       stack_keys.push(new_key);
    //       h_keys[i] = new_key;
    //       h_scores[i] = static_cast<K>(i) % freq_range;
    //       i++;
    //     }
    //     exist_keys.insert(new_key);
    //   }
    //   assert(i == key_num_append);
    // }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_append * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores_real, h_scores, key_num_append * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(key_num_append, d_keys, d_vectors, d_scores,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_append;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // For trigger the kernel selection in advance.
  int batch_size_warmup = 1;
  for (int i = 0; i < 9; i++, global_epoch++) {
    table->set_global_epoch(global_epoch);
    switch (api) {
      case API_Select::find: {
        table->find(batch_size_warmup, d_keys, d_vectors, d_found, d_scores,
                    stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::insert_or_assign: {
        table->insert_or_assign(batch_size_warmup, d_keys, d_vectors,
                                d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::find_or_insert: {
        table->find_or_insert(batch_size_warmup, d_keys, d_vectors,
                              d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::assign: {
        table->assign(batch_size_warmup, d_keys, d_def_val, d_scores,
                      stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::insert_and_evict: {
        table->insert_and_evict(batch_size_warmup, d_keys, d_vectors,
                                d_scores, d_evict_keys, d_def_val,
                                d_evict_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      case API_Select::find_ptr: {
        V** d_vectors_ptr = nullptr;
        CUDA_CHECK(
            cudaMalloc(&d_vectors_ptr, batch_size_warmup * sizeof(V*)));
        benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
                             batch_size_warmup, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        table->find(1, d_keys, d_vectors_ptr, d_found, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim,
                                 batch_size_warmup, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_vectors_ptr));
        break;
      }
      case API_Select::find_or_insert_ptr: {
        V** d_vectors_ptr = nullptr;
        bool* d_found;
        CUDA_CHECK(cudaMalloc(&d_found, batch_size_warmup * sizeof(bool)));
        CUDA_CHECK(
            cudaMalloc(&d_vectors_ptr, batch_size_warmup * sizeof(V*)));
        benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
                             batch_size_warmup, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        table->find_or_insert(batch_size_warmup, d_keys, d_vectors_ptr,
                              d_found, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_vectors_ptr));
        CUDA_CHECK(cudaFree(d_found));
        break;
      }
      case API_Select::export_batch: {
        size_t* d_dump_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));

        table->export_batch(batch_size_warmup, 0, d_dump_counter, d_keys,
                            d_vectors, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_dump_counter));
        break;
      }
      case API_Select::export_batch_if: {
        size_t* d_dump_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
        CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
        K pattern = 0;
        table->template export_batch_if<ExportIfPredFunctor>(
            pattern, threshold, batch_size_warmup, 0, d_dump_counter,
            d_keys, d_vectors, d_scores, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_dump_counter));
        break;
      }
      case API_Select::contains: {
        table->contains(1, d_keys, d_found, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        break;
      }
      default: {
        std::cout << "[Unsupport API]\n";
      }
    }
  }
  // create_keys_for_hitrate<K, S>(h_keys, h_scores, batch_size, hitrate,
  //                               Hit_Mode::last_insert, start, true /*reset*/);
  {
    int miss_count = batch_size * (1.0f - hitrate);
    uint64_t target_key_num = miss_count;
    std::unordered_set<K> numbers;
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<K> distr;
    int j = 0;

    while (numbers.size() < target_key_num) {
      numbers.insert(distr(eng));
    }
    for (const K num : numbers) {
      h_keys[j] = num;
      j++;
    }
    assert(j == miss_count);
    for (; j < batch_size; j++) {
      h_keys[j] = queue_keys.back();
      queue_keys.pop_back();
    }
    assert(j == batch_size);
  }
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, batch_size * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores_real, h_scores, batch_size * sizeof(K),
                        cudaMemcpyHostToDevice));
  auto timer = benchmark::KernelTimer<double>();
  global_epoch++;
  table->set_global_epoch(global_epoch);
  switch (api) {
    case API_Select::find: {
      timer.start();
      table->find(batch_size, d_keys, d_vectors, d_found, nullptr, stream);
      timer.end();
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(batch_size, d_keys, d_vectors, d_scores,
                              stream);
      timer.end();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(batch_size, d_keys, d_vectors, d_scores,
                            stream);
      timer.end();
      break;
    }
    case API_Select::assign: {
      timer.start();
      table->assign(batch_size, d_keys, d_def_val, d_scores, stream);
      timer.end();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      table->insert_and_evict(batch_size, d_keys, d_vectors, d_scores,
                              d_evict_keys, d_def_val, d_evict_scores, stream);
      timer.end();
      break;
    }
    case API_Select::find_ptr: {
      V** d_vectors_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&d_vectors_ptr, batch_size * sizeof(V*)));
      benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, batch_size,
                           stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.start();
      table->find(batch_size, d_keys, d_vectors_ptr, d_found, d_scores,
                  stream);
      timer.end();
      benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim, batch_size,
                               stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_vectors_ptr));
      break;
    }
    case API_Select::find_or_insert_ptr: {
      V** d_vectors_ptr = nullptr;
      bool* d_found;
      CUDA_CHECK(cudaMalloc(&d_found, batch_size * sizeof(bool)));
      CUDA_CHECK(cudaMalloc(&d_vectors_ptr, batch_size * sizeof(V*)));
      benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, batch_size,
                           stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.start();
      table->find_or_insert(batch_size, d_keys, d_vectors_ptr, d_found,
                            d_scores, stream);
      timer.end();
      CUDA_CHECK(cudaFree(d_vectors_ptr));
      CUDA_CHECK(cudaFree(d_found));
      break;
    }
    case API_Select::export_batch: {
      size_t* d_dump_counter;

      // Try to export close to but less than `batch_size` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
      timer.start();
      table->export_batch(batch_size / target_load_factor * safe_ratio, 0,
                          d_dump_counter, d_keys, d_vectors, d_scores, stream);
      timer.end();
      CUDA_CHECK(cudaFree(d_dump_counter));
      break;
    }
    case API_Select::export_batch_if: {
      size_t* d_dump_counter;

      // Try to export close to but less than `batch_size` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
      timer.start();
      K pattern = 0;
      table->template export_batch_if<ExportIfPredFunctor>(
          pattern, threshold, batch_size / target_load_factor * safe_ratio,
          0, d_dump_counter, d_keys, d_vectors, d_scores, stream);
      timer.end();
      CUDA_CHECK(cudaFree(d_dump_counter));
      break;
    }
    case API_Select::contains: {
      timer.start();
      table->contains(batch_size, d_keys, d_found, stream);
      timer.end();
      break;
    }
    default: {
      std::cout << "[Unsupport API]\n";
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores_real));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_scores));

  CUDA_CHECK(cudaDeviceSynchronize());
  CudaCheckError();

  float througput =
      batch_size / timer.getResult() / (1024 * 1024 * 1024.0f);
  return througput;
}

static Test_Mode test_mode = Test_Mode::pure_hbm;

void test_main(const uint64_t capacity = 64 * 1024 * 1024UL,
               const uint64_t batch_size = 1 * 1024 * 1024UL,
               const uint64_t hbm4values = 1024 * 1024 * 1024UL,
               const std::vector<float>& load_factors = {0.8f},
               const float hit_rate = 0.95f) {
  size_t free, total;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  if (free < hbm4values) {
    std::cerr << "free HBM is not enough, ignore current benchmark!"
              << std::endl;
    return;
  }

  TableOptions options;

  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.dim = 1;
  options.max_hbm_for_vectors = hbm4values;
  options.io_by_cpu = false;
  using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

  std::shared_ptr<Table> table = std::make_shared<Table>();
  table->init(options);

  CUDA_CHECK(cudaDeviceSynchronize());
  for (float load_factor : load_factors) {
    table->clear();
    CUDA_CHECK(cudaDeviceSynchronize());
    // There is a sampling of load_factor after several times call to target
    // API. Two consecutive calls can avoid the impact of sampling.
    auto res1 = test_one_api<Table>(table, API_Select::find, options.dim, capacity,
                                    batch_size, load_factor, hit_rate);
    auto res2 = test_one_api<Table>(table, API_Select::find, options.dim, capacity,
                                    batch_size, load_factor, hit_rate);
    std::cout << "Throughput at load factor(" << load_factor << "): " << std::max(res1, res2) << " Billion-KV/second\n";
    // std::cout << std::max(res1, res2) << std::endl;
  }
}

void print_device_prop(cudaDeviceProp& props) {
  std::cout 
    << "Device name : " << props.name << "\n"
    << "Compute Capability : " << props.major << "." << props.minor << "\n"
    // compute
    << "SM Counts : " << props.multiProcessorCount << "\n"
    << "Max Threads Number Per SM : " << props.maxThreadsPerMultiProcessor << "\n"
    << "Max Block Counts Per SM : " << props.maxBlocksPerMultiProcessor << "\n"
    << "Max Threads Number Per Block : " <<  props.maxThreadsPerBlock << "\n"
    << "Max GridSize[0 - 2] : " << props.maxGridSize[0] << " " << props.maxGridSize[1] << " " << props.maxGridSize[2] << "\n"
    << "Max ThreadsDim[0 - 2] : " << props.maxThreadsDim[0] << " " << props.maxThreadsDim[1] << " " << props.maxThreadsDim[2] << "\n"
    << "Warp Size : " << props.warpSize << "\n"
    << "Clock Rate of Device : " << props.clockRate << "\n"
    // memory
    << "Total Global Memory : " << props.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n"
    << "Max Shared Memory Per SM : " << props.sharedMemPerMultiprocessor / 1024 << " KB\n"
    << "Max Shared Memory Per Block : " << props.sharedMemPerBlock / 1024 << " KB\n"
    << "Max Registers(32bits) Number Per SM : " << props.regsPerMultiprocessor << "\n"
    << "Max Registers(32bits) Number Per Block : " << props.regsPerBlock << "\n"
    // << "Global memory bus width in bits : " << props.memoryBusWidth << "\n"
    << "Number of asynchronous engines : " << props.asyncEngineCount <<"\n"
    // cache
    // << "Memory Pitch : " << props.memPitch << "\n"
    << "Device supports caching globals in L1 : " << std::string(props.globalL1CacheSupported ? "true" : "false") << "\n"
    << "Device supports caching locals in L1 : " << std::string(props.localL1CacheSupported ? "true" : "false") << "\n"
    << "Size of L2 cache : " << props.l2CacheSize / (1024 * 1024) << " MB\n"
    << "Device's maximum l2 persisting lines capacity setting : " << props.persistingL2CacheMaxSize / (1024 * 1024) << " MB\n";
}

void print_hkv_options(const uint64_t capacity, const uint64_t batch_size, uint64_t& hbm4values, const uint64_t value_size,
    const float load_factor = 0.8f) {
  hbm4values = capacity * value_size > hbm4values ? capacity * value_size : hbm4values;
  cout << "Capacity = " << capacity / (1024 * 1024.0) << " Million-KV, "
       << "HBM = " << hbm4values / (1024 * 1024 * 1024.0) << " GB, "
       << "Batch size = " << batch_size << ", "
       << "Load factor = " << load_factor << " ";
}


int main() {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  print_device_prop(props);
  uint64_t batch_size = 1 * 1024 * 1024UL;
  cout << endl
       << "## Benchmark" << endl
       << endl
       << "* GPU: 1 x " << props.name << ": " << props.major << "."
       << props.minor << endl
       << "* Key Type = uint64_t" << endl
       << "* Value Type = uint64_t" << endl
       << "* Key-Values per OP = " << batch_size << endl
       << "* Evict strategy: LRU" << endl
       << "* `\u03BB`"
       << ": load factor" << endl
       << "* ***Throughput Unit: Billion-KV/second***" << endl
       << endl;

  try {
    test_mode = Test_Mode::pure_hbm;
    cout << "### On pure HBM mode: " << endl;

    uint64_t hbm4values = 1 * 1024 * 1024 * 1024UL;
    print_hkv_options(16 * 1024UL, 2 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(16 * 1024UL, 2 * 1024UL, hbm4values);

    print_hkv_options(128 * 1024UL, 16 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(128 * 1024UL, 16 * 1024UL, hbm4values);

    print_hkv_options(1024 * 1024UL, 128 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(1024 * 1024UL, 128 * 1024UL, hbm4values);

    print_hkv_options(16 * 1024 * 1024UL, 1024 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(16 * 1024 * 1024UL, 1024 * 1024UL, hbm4values);

    print_hkv_options(128 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(128 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values);

    print_hkv_options(1024 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(1024 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values);
  
    // load factors
    print_hkv_options(64 * 1024 * 1024UL, 1024 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(64 * 1024 * 1024UL, 1024 * 1024UL, hbm4values, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f});
    print_hkv_options(64 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values, sizeof(uint64_t));
    test_main(64 * 1024 * 1024UL, 10 * 1024 * 1024UL, hbm4values, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f});

    cout << endl;
  
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
