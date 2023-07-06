/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuda_profiler_api.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

#include "explore_kernels/common.cuh"
#include "explore_kernels/origin.cuh"
#include "explore_kernels/probing_tests.cuh"
#include "explore_kernels/copying_tests.cuh"
#include "explore_kernels/lookup_v1.cuh"
#include "explore_kernels/filter_based_lookup.cuh"
#include "explore_kernels/pipeline_general.cuh"


// #define COLLECT_PROBING_SIZE
#define PERF

using namespace nv::merlin;
using namespace benchmark;
using namespace std;

using K = uint64_t;
using S = uint64_t;
using V = float;
using TableCore_ = nv::merlin::Table<K, V, S>;
using HashTable_ = nv::merlin::HashTable<K, V, S>;
using TableOptions = nv::merlin::HashTableOptions;

/*
A100 80G:　 2039  GB/s
A100 40G:　 1555  GB/s
3090    :   936.2 GB/s
*/
constexpr float PeakBW = 2039.0f;
constexpr uint32_t REPEAT = 1;
constexpr uint32_t WARMUP = 5;
constexpr double EPSILON = 1e-3;
constexpr int PRECISION = 3;
constexpr TimeUnit tu = TimeUnit::MilliSecond;
constexpr int BLOCK_SIZE = 128;
constexpr int WARP_NUM_PER_WARP = BLOCK_SIZE / 32;

static KernelTimer<float> timer = KernelTimer<float>(tu);

template<TimeUnit tu = TimeUnit::MilliSecond>
float billionKVPerSecond(float time, uint64_t key_num_per_op = 1024 * 1024UL) {
  double billionKV = key_num_per_op / static_cast<double>(1e9);
  auto pow_ =
    static_cast<int32_t>(TimeUnit::Second) - static_cast<int32_t>(tu);
  auto factor = static_cast<float>(std::pow(10, pow_));
  float seconds = time * factor;
  return billionKV / seconds;
}

// Only considering size of value.
template<
  typename V, 
  API_Select api, 
  TimeUnit tu = TimeUnit::MilliSecond>
struct SOL;

template<typename V, API_Select api>
struct SOL<V, api, tu> {
  static float get(float time, uint64_t key_num_per_op, uint32_t dim) {
    uint64_t memory_size = static_cast<uint64_t>(
              key_num_per_op * sizeof(V) * dim * 2);
    auto pow_ =
      static_cast<int32_t>(TimeUnit::Second) - static_cast<int32_t>(tu);
    auto factor = static_cast<float>(std::pow(10, pow_));
    float seconds = time * factor;
    float curBW = static_cast<double>(memory_size) / (1024 * 1024 * 1024UL) / seconds;
    return curBW / PeakBW * 100;
  }
};

template<typename V>
struct SOL<V, API_Select::find, tu> {
  static float get(float time, uint64_t key_num_per_op, uint32_t dim, float hit_rate = 1.0f) {
    uint64_t memory_size = static_cast<uint64_t>(
              key_num_per_op * sizeof(V) * dim * 2 * hit_rate);
    auto pow_ =
      static_cast<int32_t>(TimeUnit::Second) - static_cast<int32_t>(tu);
    auto factor = static_cast<float>(std::pow(10, pow_));
    float seconds = time * factor;
    float curBW = static_cast<double>(memory_size) / (1024 * 1024 * 1024UL) / seconds;
    return curBW / PeakBW * 100;
  }
};

template<typename V>
struct SOL<V, API_Select::insert_and_evict, tu> {
  static float get(float time, uint64_t key_num_per_op, uint32_t dim, float evict_rate = 1.0f) {
    uint64_t load_store_size = key_num_per_op * sizeof(V) * dim;
    uint64_t memory_size = static_cast<uint64_t>(
              load_store_size * 4 * evict_rate + load_store_size * 2 * (1.0f - evict_rate));
    auto pow_ =
      static_cast<int32_t>(TimeUnit::Second) - static_cast<int32_t>(tu);
    auto factor = static_cast<float>(std::pow(10, pow_));
    float seconds = time * factor;
    float curBW = static_cast<double>(memory_size) / (1024 * 1024 * 1024UL) / seconds;
    return curBW / PeakBW * 100;
  }
};

template<typename K, typename V, typename S = uint64_t>
struct TestDescriptor_ {
  uint32_t bucket_size {128};
  float load_factor {1.0f};
  float hit_rate {1.0f};
  Hit_Mode hit_mode {Hit_Mode::last_insert};
  int64_t capacity {64 * 1024 * 1024UL};
  uint64_t key_num_per_op {1024 * 1024UL};
  uint32_t HBM4Values {33};  // GB
  uint32_t HMEM4Values {0};  // GB
  uint32_t dim {64};

  float getSOL(API_Select api, float elapsed_time, float rate = 1.0f) const {
    switch (api) {
      case API_Select::find:
      case API_Select::find_origin:
      case API_Select::find_origin_probing:
      case API_Select::find_tlp_probing:
      case API_Select::find_tlp_probing_8bits_uint8_t:
      case API_Select::find_tlp_probing_8bits_uint32_t:
      case API_Select::find_tlp_probing_8bits_uint4:
      case API_Select::find_tlp_probing_16bits_uint4:
      case API_Select::find_8bits_invalid_frequency:
      case API_Select::find_tlp_probing_collect_size:
      // case API_Select::find_tlp_probing_8bits_uint8_t_collect_size:
      case API_Select::find_tlp_probing_8bits_uint4_collect_size:
      case API_Select::filter_based_lookup:
      case API_Select::filter_based_lookup_prefetch:
      case API_Select::pipeline_lookup:
      case API_Select::pipeline_bucket_size:
      case API_Select::copying_origin:
      case API_Select::copying_pass_by_param:
      case API_Select::copying_multi_value:
      case API_Select::filter_lookup_prefetch_aggressively:
      case API_Select::assign: {
        return SOL<V, API_Select::find>::get(elapsed_time, key_num_per_op, 
                                             dim, hit_rate);
      }
      case API_Select::insert_and_evict: {
        return SOL<V, API_Select::insert_and_evict>::get(elapsed_time, key_num_per_op, 
                                                         dim, rate);
      }
      case API_Select::insert_or_assign:
      case API_Select::find_or_insert: {
        return SOL<V, API_Select::insert_or_assign>::get(elapsed_time, key_num_per_op, dim);        
      }
      default: {
        std::cout << "Not support get SOL at other API\n";
        return -1.0f;
      }
    }
  }
};
using TestDescriptor = TestDescriptor_<K, V, S>;

static double probing_size_avg = 0.0;
static double probing_num_avg = 0.0;
static int probing_num_max = 0;
static int probing_num_min = 0;

float test_one_api(const API_Select api, TestDescriptor& td,
                  bool sample_hit_rate = false, bool silence = true, bool check_correctness = false) {

  const uint64_t key_num_per_op = td.key_num_per_op;
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaSetDevice(0));
    init = true;
  }
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free / (1 << 30) < td.HBM4Values) {
    if (!silence) 
      std::cout << "HBM is not enough!\n";
    return -1.0f;
  }

  uint64_t key_num_init = static_cast<uint64_t>(td.capacity * td.load_factor);
  if (key_num_init < key_num_per_op && Hit_Mode::last_insert == td.hit_mode) {
    if (!silence) 
      std::cout << "Keys for init is too few!\n";
    return -1.0f;
  }

  TableOptions options;
  options.init_capacity = td.capacity;
  options.max_capacity = td.capacity;
  options.dim = td.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.io_by_cpu = false;
  options.evict_strategy = EvictStrategy::kCustomized;
  options.max_bucket_size = td.bucket_size;

  std::unique_ptr<HashTable_> table = std::make_unique<HashTable_>();
  table->init(options);

  TableCore_* table_core = table->get_host_table();
  TableCore_* table_core_device = table->get_device_table();

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

  CUDA_CHECK(
      cudaMemset(h_vectors, 0, key_num_per_op * sizeof(V) * options.dim));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  V* d_vect_contrast;
  bool* d_found;
  K* d_evict_keys;
  S* d_evict_scores;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMalloc(&d_vect_contrast, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_evict_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_scores, key_num_per_op * sizeof(S)));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemset(d_vect_contrast, 2, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const float target_load_factor = key_num_init * 1.0f / td.capacity;
  uint64_t key_num_remain = key_num_init % key_num_per_op == 0
                                ? key_num_per_op
                                : key_num_init % key_num_per_op;
  int32_t loop_num_init = (key_num_init + key_num_per_op - 1) / key_num_per_op;

  // no need to get load factor
  K start = 0UL;
  for (int i = 0; i < loop_num_init; i++) {
    uint64_t key_num_cur_insert =
        i == loop_num_init - 1 ? key_num_remain : key_num_per_op;

    create_continuous_keys<K, S>(h_keys, h_scores, key_num_cur_insert, start);
    if (check_correctness) {
      init_value_using_key<K, V>(h_keys, h_vectors, key_num_cur_insert, options.dim);
      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                            key_num_cur_insert * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_cur_insert * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_cur_insert * sizeof(S),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(key_num_cur_insert, d_keys, d_vectors, d_scores,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_cur_insert;
  }
  if (!silence)
    std::cout << "Loop number for init : " << loop_num_init << std::endl;

  // read_load_factor <= target_load_factor always true, due to evict occurrence
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * td.capacity);
    if (key_num_append <= 0) break;
    if (key_num_append > key_num_per_op) key_num_append = key_num_per_op;

    // if (!silence)
    //   std::cout << "Extra insert keys : " << key_num_append << std::endl;

    create_continuous_keys<K, S>(h_keys, h_scores, key_num_append, start);
    if (check_correctness) {
      init_value_using_key<K, V>(h_keys, h_vectors, key_num_append, options.dim);
      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                            key_num_append * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_append * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_append * sizeof(S),
                          cudaMemcpyHostToDevice));

    table->insert_or_assign(key_num_append, d_keys, d_vectors, d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_append;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  if (!silence)
    std::cout << "Load factor : " << fixed << setprecision(PRECISION)
              << real_load_factor << std::endl;

  create_keys_for_hitrate<K, S>(h_keys, h_scores, key_num_per_op, td.hit_rate,
                                        td.hit_mode, start, true /*reset*/);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S),
                        cudaMemcpyHostToDevice));
  if (sample_hit_rate && api != API_Select::insert_and_evict) {
    table->find(key_num_per_op, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_found, d_found, key_num_per_op * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    int found_num = 0;
    for (int i = 0; i < key_num_per_op; i++) {
      if (h_found[i]) {
        found_num++;
      }
    }
    std::cout << "Hit rate : " << 1.0 * found_num / key_num_per_op << std::endl;
  }

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  V** values_addr;
  CUDA_CHECK(cudaMalloc(&values_addr, key_num_per_op * sizeof(V*)));
  int * times_host;
  CUDA_CHECK(cudaMallocHost(&times_host, sizeof(int) * key_num_per_op));
  
  size_t evicted_number = 0;

  cudaProfilerStart();
  switch (api) {
    case API_Select::find_origin: {
      timer.start();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, d_vectors, d_scores, d_found, N); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, d_vectors, d_scores, d_found, N);
      }
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find_origin_probing: {
      timer.start();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin_probing<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, values_addr, d_scores, d_found, N); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin_probing<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, values_addr, d_scores, d_found, N);
      }
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;      
    }
    case API_Select::find_tlp_probing: {
      timer.start();
      lookup_kernel_with_io_tlp_probing<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;       
    }
    case API_Select::find_tlp_probing_8bits_uint8_t: {
      timer.start();
      //--------------- probing using 8 bits digests(load 8 bits every time) --------------
      lookup_kernel_with_io_tlp_probing_8bits_uint8_t<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find_tlp_probing_8bits_uint32_t: {
      timer.start();
      //--------------- probing using 8 bits digests(load 32 bits every time) --------------
      lookup_kernel_with_io_tlp_probing_8bits_uint32_t<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find_tlp_probing_8bits_uint4: {
      timer.start();
      //------------------ probing using 8 bits digests(load uint4 every time)------------
      lookup_kernel_with_io_tlp_probing_8bits_uint4<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find_tlp_probing_16bits_uint4: {
      timer.start();
      //-------------------probing using 16 bits digests(load uint4 every time) -----------
      lookup_kernel_with_io_tlp_probing_16bits_uint4<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.end();
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find_8bits_invalid_frequency: {
      int* times_dev;
      cudaMalloc(&(times_dev), sizeof(int) * key_num_per_op);
      cudaMemset(times_dev, 0, key_num_per_op * sizeof(int));

      lookup_kernel_with_io_8bits_invalid_frequency<K, V, S>
        <<<key_num_per_op / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table_core->buckets, table_core->buckets_num, options.dim,
          d_keys, values_addr, d_scores, d_found, key_num_per_op, times_dev);
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaMemcpy(times_host, times_dev, sizeof(int) * key_num_per_op, cudaMemcpyDeviceToHost);
      std::vector<int> invalid_frequency(512, 0);
      for (int i = 0; i < key_num_per_op; i++) {
        invalid_frequency[times_host[i]] += 1;
      }
      int end = 511;
      for (int i = 511; i >=0; i--) {
        if (invalid_frequency[i] != 0) break;
        end = i;
      }
      end = end < 20 ? 20 : end;
      for (int i = 0; i < end; i++) {
        std::cout << "Invalid times " << i << "\t"
                  << "Frequency " << invalid_frequency[i] << std::endl;
      }
      CUDA_CHECK(cudaFree(times_dev));
      break;
    }
    case API_Select::find_tlp_probing_collect_size: {
      int* times_dev;
      cudaMalloc(&(times_dev), sizeof(int) * key_num_per_op);
      cudaMemset(times_dev, 0, key_num_per_op * sizeof(int));

      lookup_kernel_with_io_tlp_probing_collect_size<K, V, S>
        <<<key_num_per_op / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table_core->buckets, table_core->buckets_num, options.dim,
          d_keys, values_addr, d_scores, d_found, key_num_per_op, times_dev);
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaMemcpy(times_host, times_dev, sizeof(int) * key_num_per_op, cudaMemcpyDeviceToHost);
      size_t sum = 0;
      probing_num_max = 0;
      probing_num_min = 129;
      for (int i = 0; i < key_num_per_op; i++) {
        sum += times_host[i];
        if (probing_num_max < times_host[i]) probing_num_max = times_host[i];
        if (probing_num_min > times_host[i]) probing_num_min = times_host[i];
      }
      sum *= sizeof(K);
      probing_size_avg = static_cast<double>(sum) / key_num_per_op;
      probing_num_avg = probing_size_avg / sizeof(K);
      CUDA_CHECK(cudaFree(times_dev));
      break;
    }
    case API_Select::find_tlp_probing_8bits_uint4_collect_size: {
      int* times_dev;
      cudaMalloc(&(times_dev), sizeof(int) * key_num_per_op);
      cudaMemset(times_dev, 0, key_num_per_op * sizeof(int));

      lookup_kernel_with_io_tlp_probing_8bits_uint4_collect_size<K, V, S>
        <<<key_num_per_op / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table_core->buckets, table_core->buckets_num, options.dim,
          d_keys, values_addr, d_scores, d_found, key_num_per_op, times_dev);
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_core_origin_copying<V, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              d_vectors, values_addr, d_found, options.dim, key_num_per_op); 
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaMemcpy(times_host, times_dev, sizeof(int) * key_num_per_op, cudaMemcpyDeviceToHost);
      size_t sum = 0;
      probing_num_max = 0;
      probing_num_min = 129;
      for (int i = 0; i < key_num_per_op; i++) {
        sum += times_host[i];
        if (probing_num_max < times_host[i]) probing_num_max = times_host[i];
        if (probing_num_min > times_host[i]) probing_num_min = times_host[i];
      }
      sum *= sizeof(K);
      probing_size_avg = static_cast<double>(sum) / key_num_per_op;
      probing_num_avg = probing_size_avg / sizeof(K);
      CUDA_CHECK(cudaFree(times_dev));
      break;
    }
    case API_Select::filter_based_lookup: {
      timer.start();
      ////////////////////////////////////////////////////////////////////////////////////
      if (options.dim == 4) 
        lookup_kernel_with_io_v4_1<K, V, S, 1>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      else if (options.dim == 64)
        lookup_kernel_with_io_v4_1<K, V, S, 16>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      else if (options.dim == 128)
        lookup_kernel_with_io_v4_1<K, V, S, 16>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::filter_based_lookup_prefetch: {
      timer.start();
      ////////////////////////////////////////////////////////////////////////////////////
      if (options.dim == 4) 
        lookup_kernel_with_io_v4_2<K, V, S, 1, 4>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      else if (options.dim == 64)
        lookup_kernel_with_io_v4_2<K, V, S, 16, 64>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      else if (options.dim == 128)
        lookup_kernel_with_io_v4_2<K, V, S, 32, 128>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::pipeline_lookup: {
      timer.start();
      /////--------------------------------------------------------------------------------
      using CopyScore = CopyScoreByPassCache<S, K, 128>;
      // using CopyScore = CopyScoreEmpty<S, K, 128>;
      if (options.dim == 4) {
        using CopyValue = CopyValueOneGroup<float, float4, 16>;
        lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue, 128>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      } else if (options.dim == 64) {
        using CopyValue = CopyValueOneGroup<float, float4, 16>;
        lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue, 128>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      } else if (options.dim == 128) {
        using CopyValue = CopyValueTwoGroup<float, float4, 16>;
        lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue, 128>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
                d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      }
      //////////////////////////////////////////////////////////////////////////////////////
      // lookup_kernel_with_io_v1<K, V, S>
      //     <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //         table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //         d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::pipeline_bucket_size: {
      // timer.start();
      // if (td.bucket_size == 256) {
      //   using CopyScore = CopyScoreByPassCache<S, K, 256>;
      //   if (options.dim == 4) {
      //     using CopyValue = CopyValueOneGroup<float, float4, 64>;
      //     lookup_kernel_with_io_pipeline_v3<K, float, S, 128, 64, 256, CopyScore, CopyValue, 4>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //             table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //             d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      //   }
      // } else if (td.bucket_size == 512) {
      //   using CopyScore = CopyScoreByPassCache<S, K, 512>;
      //   if (options.dim == 4) {
      //     using CopyValue = CopyValueOneGroup<float, float4, 128>;
      //     lookup_kernel_with_io_pipeline_v3<K, float, S, 128, 128, 512, CopyScore, CopyValue, 4>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //             table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //             d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      //   }
      // } else if (td.bucket_size == 1024) {
      //   using CopyScore = CopyScoreByPassCache<S, K, 1024>;
      //   if (options.dim == 4) {
      //     using CopyValue = CopyValueOneGroup<float, float4, 256>;
      //     lookup_kernel_with_io_pipeline_v3<K, float, S, 256, 256, 1024, CopyScore, CopyValue, 4>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //             table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //             d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      //   }
      // } else if (td.bucket_size == 2048) {
      //   using CopyScore = CopyScoreByPassCache<S, K, 2048>;
      //   if (options.dim == 4) {
      //     using CopyValue = CopyValueOneGroup<float, float4, 512>;
      //     lookup_kernel_with_io_pipeline_v3<K, float, S, 512, 512, 2048, CopyScore, CopyValue, 4>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //             table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //             d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      //   }
      // }
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      // timer.end();
      // break;
    }
    case API_Select::copying_origin: {
      lookup_kernel_with_io_tlp_probing_8bits_uint4<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      // --------------------------- copy value using single kernel ---------------------
      timer.start();
      if (options.dim == 4) {
        lookup_kernel_with_io_copying_origin<K, V, S, 1, 0>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core_device, d_vectors, values_addr, d_found, key_num_per_op);
      } else if (options.dim == 128) {
        lookup_kernel_with_io_copying_origin<K, V, S, 32, 5>
            <<<(key_num_per_op + WARP_NUM_PER_WARP - 1) / WARP_NUM_PER_WARP, BLOCK_SIZE, 0, stream>>>(
              table_core_device, d_vectors, values_addr, d_found, key_num_per_op);
      }
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::copying_pass_by_param: {
      lookup_kernel_with_io_tlp_probing_8bits_uint4<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      // --------------------------- copy value using single kernel ---------------------
      timer.start();
      if (options.dim == 4) {
        lookup_kernel_with_io_v2_kernel2<K, V, S, 1, 0>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      } else if (options.dim == 128) {
        lookup_kernel_with_io_v2_kernel2<K, V, S, 32, 5>
            <<<(key_num_per_op + WARP_NUM_PER_WARP - 1) / WARP_NUM_PER_WARP, BLOCK_SIZE, 0, stream>>>(
              options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      }
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::copying_multi_value: {
      lookup_kernel_with_io_tlp_probing_8bits_uint4<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      // --------------------------- copy value using single kernel ---------------------
      timer.start();
      //--------------------------- copy values using test1 ---------------------
      if (options.dim == 4) {
        lookup_kernel_with_io_v2_kernel2_test1<K, V, S, 1, 0>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      } else if (options.dim == 128) {
        lookup_kernel_with_io_v2_kernel2_test1<K, V, S, 32, 5>
             <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      }
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::filter_lookup_prefetch_aggressively: {
      timer.start();
      if (options.dim == 4) {
        lookup_kernel_with_io_filter<K, V, S, float4, 1, 64>
            <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, (int)(options.max_bucket_size), table_core->buckets_num, static_cast<int>(options.dim/4), 
              d_keys, reinterpret_cast<float4*>(d_vectors), d_scores, d_found, key_num_per_op);
      } else if (options.dim == 128) {
        lookup_kernel_with_io_filter<K, V, S, float4, 32, 64>
             <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, (int)(options.max_bucket_size), table_core->buckets_num, static_cast<int>(options.dim/4), 
              d_keys, reinterpret_cast<float4*>(d_vectors), d_scores, d_found, key_num_per_op);
      }
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::find: {
      timer.start();
      //////////////////////////////////////////////////////////////////////////////////////
      // origin
      if (td.load_factor <= 0.75) {
        const unsigned int tile_size = 4;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, d_vectors, d_scores, d_found, N); 
      } else {
        const unsigned int tile_size = 16;
        const size_t N = key_num_per_op * tile_size;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, BLOCK_SIZE);
        lookup_kernel_with_io_origin<K, V, S, tile_size>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
              table_core_device, options.max_bucket_size, table_core->buckets_num, options.dim,
              d_keys, d_vectors, d_scores, d_found, N);
      }
      //////////////////////////////////////////////////////////////////////////////////////
      // table->find(key_num_per_op, d_keys, d_vectors, d_found, d_scores, stream);
      ///////////////////////////////// copy value--------------------------------------
      // timer.start();
      //--------------------------- copy values using test2 ---------------------
      // if (check_correctness) {
      //   if (options.dim == 4) {
      //     lookup_kernel_with_io_v2_kernel2_test2<K, V, S, 1, 0>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //           options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      //   } else if (options.dim == 128) {
      //     lookup_kernel_with_io_v2_kernel2_test2<K, V, S, 32, 5>
      //         <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //           options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      //   }
      // }
      //////////////////////////////////////////////////////////////////////////////////////

      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(key_num_per_op, d_keys, d_vectors, d_scores,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::assign: {
      timer.start();
      table->assign(key_num_per_op, d_keys, d_vect_contrast, d_scores, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      evicted_number = table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_scores,
                              d_evict_keys, d_vect_contrast, d_evict_scores,
                              stream);
      timer.end();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      break;
    }
    default: {
      std::cout << "Unsupport API!\n";
    }
  }
  cudaProfilerStop();
  
  if (api == API_Select::insert_and_evict)
    td.hit_rate = evicted_number / key_num_per_op;

  if (check_correctness) {
    switch (api) {
      case API_Select::find_origin:
      case API_Select::find_origin_probing:
      case API_Select::find_tlp_probing:
      case API_Select::find_tlp_probing_8bits_uint8_t:
      case API_Select::find_tlp_probing_8bits_uint32_t:
      case API_Select::find_tlp_probing_8bits_uint4:
      case API_Select::find_tlp_probing_16bits_uint4:
      case API_Select::find_8bits_invalid_frequency:
      case API_Select::find_tlp_probing_collect_size:
      case API_Select::find_tlp_probing_8bits_uint8_t_collect_size:
      case API_Select::find_tlp_probing_8bits_uint4_collect_size:
      case API_Select::filter_based_lookup:
      case API_Select::filter_based_lookup_prefetch:
      case API_Select::pipeline_lookup:
      case API_Select::pipeline_bucket_size:
      case API_Select::copying_origin:
      case API_Select::copying_pass_by_param:
      case API_Select::copying_multi_value:
      case API_Select::filter_lookup_prefetch_aggressively:
      case API_Select::find: {
        bool* h_found;
        CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));
        CUDA_CHECK(cudaMemcpy(h_found, d_found, key_num_per_op * sizeof(bool),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                              key_num_per_op * sizeof(V) * options.dim,
                              cudaMemcpyDeviceToHost));
        int found_num = 0;
        int correct_value = 0;
        for (int i = 0; i < key_num_per_op; i++) {
          if (h_found[i]) {
            found_num++;
            bool correct = true;
            for (int j = 0; j < options.dim; j++) {
              if (h_vectors[i * options.dim + j] !=
                  static_cast<float>(h_keys[i] * 0.00001)) {
                correct = false;
                break;
              }
            }
            if (correct) {
              correct_value += 1;
            }
          }
        }
        std::cout << "Found keys : " << found_num << std::endl;
        std::cout << "Correct value : " << correct_value << std::endl;
        CUDA_CHECK(cudaFreeHost(h_found));
        break;
      }
      case API_Select::insert_or_assign: {
        break;
      }
      case API_Select::find_or_insert: {
        break;
      }
      case API_Select::assign: {
        break;
      }
      case API_Select::insert_and_evict: {
        break;
      }
      default: {
        std::cout << "Unsupport API!\n";
      }
    }
  }

  uint32_t hmem4values =
      td.capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024UL);
  hmem4values = hmem4values < td.HBM4Values ? 0 : (hmem4values - td.HBM4Values);
  td.HMEM4Values = hmem4values;

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_vect_contrast));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_scores));
  CUDA_CHECK(cudaFree(values_addr));

  CUDA_CHECK(cudaFreeHost(times_host));

  CudaCheckError();

  return timer.getResult();
}

template<typename Rep>
Rep getAverage(const std::vector<Rep>& arrs) {
  Rep sum = static_cast<Rep>(0.0f);
  int effective_num = 0;
  for (auto const & item : arrs) {
    if (item > 0) {
      sum += item;
      effective_num += 1;
    }
  }
  return static_cast<Rep>(sum / effective_num);
}

void test_cudaMemcpyAsync(const TestDescriptor& td) {
  for (int i = 0; i < WARMUP; i++) {
    test_cudaMemcpyAsync<V, REPEAT>(td.dim * td.key_num_per_op);
  }
  for (int i = 0; i < 20; i++) {
    std::vector<float> elapsed_time(REPEAT, 0.0f);
    for (int i = 0; i < REPEAT; i++) {
      elapsed_time.emplace_back(test_cudaMemcpyAsync<V, REPEAT>(td.dim * td.key_num_per_op));
    }
    float average_time = getAverage(elapsed_time);
    float sol = td.getSOL(API_Select::find, average_time);
    std::cout << "cudaMemcpyAsync's SOL: "
              << average_time << " ms \t\t" 
              << sol << " %\n"; 
  }
}

void test_memcpy_kernel(const TestDescriptor& td) {
  for (int i = 0; i < WARMUP; i++) {
    test_memcpy_kernel<V, REPEAT>(td.dim * td.key_num_per_op);
  }
  for (int i = 0; i < 20; i++) {
    std::vector<float> elapsed_time(REPEAT, 0.0f);
    for (int i = 0; i < REPEAT; i++) {
      elapsed_time.emplace_back(test_memcpy_kernel<V, REPEAT>(td.dim * td.key_num_per_op));
    }
    float average_time = getAverage(elapsed_time);
    float sol = td.getSOL(API_Select::find, average_time);
    std::cout << "memcpy kernel's SOL: "
              << average_time << " ms \t\t" 
              << sol << " %\n"; 
  }
}

#include "explore_kernels/bucket_size.cuh"

void test_bucket_size() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;
  td.dim = 4;
  td.key_num_per_op = 256UL;

  two_lines();
  std::cout << "Test the bucket size's effect to Hashtable\n";
  two_lines();

  one_line();
  std::cout << "Test correct rate for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\n";
  one_line();

  for (int bkt_size = 8; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    std::cout << "Bucket size " << bkt_size << "\n";
    test_evict_strategy_customized_correct_rate(td);
  }


  one_line();
  std::cout << "Test the load factor when the first evict occurs for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\n";
  one_line();

  for (int bkt_size = 8; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    size_t evict_number = 0;
    float load_factor = test_when_evict_occur(td, evict_number);
    std::cout << "Bucket size " << bkt_size << "\t\t"
              << "Load factor " << load_factor << "\t\t"
              << "Evict_number " << evict_number << "\n";
  }

  one_line();
  std::cout << "Test the evict number when table is full for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\n";
  one_line();

  for (int bkt_size = 8; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    size_t evict_number = 0;
    float load_factor = test_evict_number_when_bucket_full(td, evict_number);
    std::cout << "Bucket size " << bkt_size << "\t\t"
              << "Load factor " << load_factor << "\t\t"
              << "Evict_number " << evict_number << "\n";
  }

  one_line();
  std::cout << "Test the max score be evicted in the load factor interval for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\n";
  one_line();

  for (int bkt_size = 8; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    one_line();
    std::cout << "Bucket size = " << td.bucket_size << "\n";
    one_line();
    test_max_score_evicted_per_interval(td);
  }


  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  one_line();
  std::cout << "Test probing time when hashtable is full for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\t"
            << "dim " << td.dim * sizeof(V) << " B\n";
  one_line();

  for (int bkt_size = 8; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    // float elapse_time = test_one_api(API_Select::find_origin, td, false, true, true);
    float elapse_time = test_one_api(API_Select::find_origin, td);
    std::cout << "Bucket size = " << td.bucket_size << "\t\t"
              << "Elapse time " << elapse_time << " ms\n";
  }
}

void test_by_discription(TestDescriptor& td, API_Select api, bool check = false) {
  // Check correctness
  td.load_factor = 1.0f;
  test_one_api(api, td, true, false, true);
  // Warpup
  for (int i = 0; i < WARMUP; i++) {
    test_one_api(api, td);
  }
  bool sample_hit_rate = api == API_Select::insert_and_evict ? true : false;
  for (float lf = 0.02; lf < 1.0f; lf += 0.03) {
    td.load_factor = lf;
    std::vector<float> elapsed_time(REPEAT, 0.0f);
    for (int i = 0; i < REPEAT; i++) {
      if (check) {
        one_line();
        elapsed_time.emplace_back(test_one_api(api, td, sample_hit_rate, true, true));
      }
      else
        elapsed_time.emplace_back(test_one_api(api, td, sample_hit_rate));
    }
    float average_time = getAverage(elapsed_time);
    float tput = billionKVPerSecond(average_time, td.key_num_per_op);
    float sol = td.getSOL(api, average_time, td.hit_rate);
    
    std::cout << "Load factor " << lf << "\t"
              << average_time << " ms \t\t" 
              << tput << " billionKV/s \t\t" 
              << sol << " %\n"; 
  }
}


void test_find_origin() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test find, origin implement\n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::find_origin);

  td.dim = 128;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  test_by_discription(td, API_Select::find_origin);
}

void test_find_origin_probing() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test find, origin, probing implement\n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::find_origin_probing);
}

void test_find_probing_tlp() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test find, thread level parallelism, probing\n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::find_tlp_probing);
}

void test_find_probing_tlp_digest() {

  std::vector<API_Select> apis(4);
  apis[0] = API_Select::find_tlp_probing_8bits_uint8_t;
  apis[1] = API_Select::find_tlp_probing_8bits_uint32_t;
  apis[2] = API_Select::find_tlp_probing_8bits_uint4;
  apis[3] = API_Select::find_tlp_probing_16bits_uint4;

  std::vector<std::string> infos {
    "8bits loading uint8_t",
    "8bits loading uint32_t",
    "8bits loading uint4",
    "16bits loading uint4"
  };

  for (int i = 0; i < 4; i++) {
    auto api = apis[i];

    TestDescriptor td;
    td.capacity = 64 * 1024 * 1024UL;
    td.key_num_per_op = 1024 * 1024UL;
    td.load_factor = 1.0f;
    td.hit_rate = 1.0f;
    td.hit_mode = Hit_Mode::last_insert;

    two_lines();
    std::cout << "Test find, thread level parallelism, probing, " << infos[i] << "\n";
    two_lines();

    td.dim = 4;
    one_line();
    std::cout << "Capacity = 64 * 1024 * 1024 \t"
              << "Batch = 1024 * 1024 \t"
              << "Key = 64 bits \t"
              << "Value = 16 B\n";
    one_line();
    test_by_discription(td, api);
  }
}

void test_digest_invalid_frequency() {

  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test find, 8 bits digests, frequency of invalid times\n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  td.load_factor = 0.2f;
  test_one_api(API_Select::find_8bits_invalid_frequency, td, true, false, true);
  for (float lf = 0.02; lf < 1.0f; lf += 0.03) {
    td.load_factor = lf;
    one_line();
    std::cout << "Load factor " << lf << "\n";
    one_line();
    test_one_api(API_Select::find_8bits_invalid_frequency, td);
  }
}

void test_collect_probing_size() {

  std::vector<API_Select> apis(2);
  apis[0] = API_Select::find_tlp_probing_collect_size;
  apis[1] = API_Select::find_tlp_probing_8bits_uint4_collect_size;

  std::vector<std::string> infos {
    "orginal key, collect probing memory size",
    "8bits loading uint4, collect probing memory size"
  };

  for (int i = 0; i < 2; i++) {
    auto api = apis[i];

    TestDescriptor td;
    td.capacity = 64 * 1024 * 1024UL;
    td.key_num_per_op = 1024 * 1024UL;
    td.load_factor = 1.0f;
    td.hit_rate = 1.0f;
    td.hit_mode = Hit_Mode::last_insert;

    two_lines();
    std::cout << "Test find, thread level parallelism, probing, " << infos[i] << "\n";
    two_lines();

    td.dim = 4;
    one_line();
    std::cout << "Capacity = 64 * 1024 * 1024 \t"
              << "Batch = 1024 * 1024 \t"
              << "Key = 64 bits \t"
              << "Value = 16 B\n";
    one_line();
    test_one_api(api, td, true, false, true);
    std::cout << "Load factor " << td.load_factor << "\t" 
              << "Probing <size> AVG " << probing_size_avg << "\t"
              << "<num> AVG " << probing_num_avg  << "\t "
              << "<num> MAX " << probing_num_max  << "\t "
              << "<num> MIN " << probing_num_min <<  std::endl;
  }

}

void test_filter_based_lookup() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test filter based lookup \n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::filter_based_lookup);

  td.dim = 128;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  test_by_discription(td, API_Select::filter_based_lookup);
}

void test_filter_based_lookup_prefetch() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test filter based lookup prefetch \n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::filter_based_lookup_prefetch);

  td.dim = 128;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  test_by_discription(td, API_Select::filter_based_lookup_prefetch);
}

void test_pipiline_based_lookup() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test pipeline lookup \n";
  two_lines();

  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  test_by_discription(td, API_Select::pipeline_lookup);

  td.dim = 128;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  test_by_discription(td, API_Select::pipeline_lookup);
}

void test_speedup() {


  auto contrast_test = [](TestDescriptor& td, API_Select api) {
    for (int i = 0; i < WARMUP; i++) {
      test_one_api(api, td);
    }
    constexpr int REPEAT_LOCAL = 10;
    std::vector<float> elapsed_time(REPEAT_LOCAL, 0.0f);
    for (int i = 0; i < REPEAT_LOCAL; i++) {
      elapsed_time[i] = test_one_api(api, td);
    }
    float average_time = getAverage(elapsed_time);
    float tput = billionKVPerSecond(average_time, td.key_num_per_op);
    float sol = td.getSOL(api, average_time);
    
    std::cout << average_time << " ms \t\t" 
              << tput << " billionKV/s \t\t" 
              << sol << " %\n";
    one_line();
    // test correctness
    test_one_api(api, td, true, false, true);
    one_line();
  };

  two_lines();
  std::cout << "Test Speedup \n";
  two_lines();

  TestDescriptor td;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.dim = 128;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);

  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.dim = 64;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 256 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);

  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.dim = 4;
  one_line();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);

  td.capacity = 15 * 1024 * 1024UL;
  td.key_num_per_op = 256 * 1024UL;
  td.dim = 128;
  one_line();
  std::cout << "Capacity = 15 * 1024 * 1024 \t"
            << "Batch = 256 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);

  td.capacity = 15 * 1024 * 1024UL;
  td.key_num_per_op = 256 * 1024UL;
  td.dim = 64;
  one_line();
  std::cout << "Capacity = 15 * 1024 * 1024 \t"
            << "Batch = 256 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 256 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);

  td.capacity = 15 * 1024 * 1024UL;
  td.key_num_per_op = 256 * 1024UL;
  td.dim = 4;
  one_line();
  std::cout << "Capacity = 15 * 1024 * 1024 \t"
            << "Batch = 256 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  one_line();
  std::cout << "find origin \t\t";
  contrast_test(td, API_Select::find_origin);
  std::cout << "filter lookup \t\t";
  contrast_test(td, API_Select::filter_based_lookup);
  std::cout << "pipeline lookup \t";
  contrast_test(td, API_Select::pipeline_lookup);
}

void test_pipeline_bucket_size() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;
  td.dim = 4;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;

  one_line();
  std::cout << "Test pipeline_bucket_size when hashtable is full for different bucket size\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\t"
            << "dim " << td.dim * sizeof(V) << " B\n";
  one_line();

  for (int bkt_size = 256; bkt_size < 4096; bkt_size *= 2) {
    td.bucket_size = bkt_size;
    test_one_api(API_Select::pipeline_bucket_size, td, false, true, true);
    float elapse_time = test_one_api(API_Select::pipeline_bucket_size, td);
    std::cout << "Bucket size = " << td.bucket_size << "\t\t"
              << "Elapse time " << elapse_time << " ms\n";
  }
}

void test_copying_kernel() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;
  td.dim = 128;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;

  one_line();
  std::cout << "Test copying kernel\n";
  std::cout << "Capacity: " << td.capacity / 1024 / 1024 << " * 1024 * 1024\t\t"
            << "key number per operation: " << td.key_num_per_op << "\t"
            << "dim " << td.dim * sizeof(V) << " B\n";
  one_line();

  // test_by_discription(td, API_Select::copying_origin);
  test_by_discription(td, API_Select::copying_pass_by_param);
  test_by_discription(td, API_Select::copying_multi_value);
  // test_by_discription(td, API_Select::copying_multi_value_prefetch);
}

void test_lookup_filter_prefetch_aggressively() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test filter lookup (prefetch aggressively)\n";
  two_lines();

  td.dim = 4;
  two_lines();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  two_lines();
  test_by_discription(td, API_Select::filter_lookup_prefetch_aggressively);

  td.dim = 128;
  two_lines();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  two_lines();
  test_by_discription(td, API_Select::filter_lookup_prefetch_aggressively);
}

void test_insert_and_evict() {
  TestDescriptor td;
  td.capacity = 64 * 1024 * 1024UL;
  td.key_num_per_op = 1024 * 1024UL;
  td.load_factor = 1.0f;
  td.hit_rate = 1.0f;
  td.hit_mode = Hit_Mode::last_insert;

  two_lines();
  std::cout << "Test insert_and_evict: 1 thread probing, group reduction, copying\n";
  two_lines();

  td.dim = 4;
  two_lines();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 16 B\n";
  two_lines();
  test_by_discription(td, API_Select::insert_and_evict);

  td.dim = 128;
  two_lines();
  std::cout << "Capacity = 64 * 1024 * 1024 \t"
            << "Batch = 1024 * 1024 \t"
            << "Key = 64 bits \t"
            << "Value = 512 B\n";
  two_lines();
  test_by_discription(td, API_Select::insert_and_evict);
}

int main(int argc, char* argv[]) {
  try {
    {
      // test_bucket_size();

      // test_find_origin();

      // test_find_origin_probing();

      // test_find_probing_tlp();

      // test_find_probing_tlp_digest();

      // test_digest_invalid_frequency();

      // test_collect_probing_size();

      // test_filter_based_lookup();
      // test_filter_based_lookup_prefetch();

      // test_pipiline_based_lookup();

      //  test_speedup();

      ///TODO:
      // test_pipeline_bucket_size();

      // test_copying_kernel();

      // test_lookup_filter_prefetch_aggressively();

      test_insert_and_evict();


      // TestDescriptor td;
      // td.capacity = 64 * 1024 * 1024UL;
      // td.key_num_per_op = 1024 * 1024UL;
      // td.load_factor = 1.0f;
      // td.hit_rate = 1.0f;
      // td.hit_mode = Hit_Mode::last_insert;
      // td.dim = 128;
      // test_cudaMemcpyAsync(td);
      // test_memcpy_kernel(td);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}