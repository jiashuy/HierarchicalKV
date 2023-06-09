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
#include "explore_kernels/lookup_v1.cuh"
#include "explore_kernels/lookup_v2.cuh"

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
constexpr uint32_t REPEAT = 2;
constexpr uint32_t WARMUP = 10;
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
  if (sample_hit_rate) {
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


  cudaProfilerStart();
  switch (api) {
    case API_Select::find: {
      // timer.start();
      //////////////////////////////////////////////////////////////////////////////////////
      // table->find(key_num_per_op, d_keys, d_vectors, d_found, nullptr, stream);
      //////////////////////////////////////////////////////////////////////////////////////
      // lookup_kernel_with_io_v1<K, V, S>
      //     <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //         table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
      //         d_keys, d_vectors, d_scores, d_found, key_num_per_op);
      //////////////////////////////////////////////////////////////////////////////////////
      //--------------------------- probing using single kernel ---------------------
      lookup_kernel_with_io_v2_kernel1<K, V, S>
          <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
              table_core->buckets, table_core->buckets_num, static_cast<int>(options.dim), 
              d_keys, values_addr, d_scores, d_found, key_num_per_op);
      timer.start();
      ////////////////////////////////////////////--------------------------------------
      //--------------------------- copy value using single kernel ---------------------
      // if (options.dim == 4) {
      //   lookup_kernel_with_io_v2_kernel2<K, V, S, 1, 0>
      //       <<<(key_num_per_op + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      //         options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      // } else if (options.dim == 128) {
      //   lookup_kernel_with_io_v2_kernel2<K, V, S, 32, 5>
      //       <<<(key_num_per_op + WARP_NUM_PER_WARP - 1) / WARP_NUM_PER_WARP, BLOCK_SIZE, 0, stream>>>(
      //         options.dim, d_vectors, values_addr, d_found, key_num_per_op);
      // }
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
      //////////////////////////////////////////////////////////////////////////////////////

      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
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
      table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_scores,
                              d_evict_keys, d_vect_contrast, d_evict_scores,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    default: {
      std::cout << "Unsupport API!\n";
    }
  }
  cudaProfilerStop();

  if (check_correctness) {
    switch (api) {
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

void test_by_discription(TestDescriptor& td) {
  // Check correctness
  test_one_api(API_Select::find, td, true, false, true);
  // Warpup
  for (int i = 0; i < WARMUP; i++) {
    test_one_api(API_Select::find, td);
  }
  for (float lf = 0.02; lf < 1.0f; lf += 0.03) {
    td.load_factor = lf;
    std::vector<float> elapsed_time(REPEAT, 0.0f);
    for (int i = 0; i < REPEAT; i++) {
      elapsed_time.emplace_back(test_one_api(API_Select::find, td));
    }
    float average_time = getAverage(elapsed_time);
    float tput = billionKVPerSecond(average_time, td.key_num_per_op);
    float sol = td.getSOL(API_Select::find, average_time);
    std::cout << "Load factor " << lf << "\t"
              << average_time << " ms \t\t" 
              << tput << " billionKV/s \t\t" 
              << sol << " %\n"; 
  }
}

int main(int argc, char* argv[]) {
  try {
    {
      TestDescriptor td;
      td.capacity = 64 * 1024 * 1024UL;
      td.key_num_per_op = 1024 * 1024UL;
      td.load_factor = 1.0f;
      td.hit_rate = 1.0f;
      td.hit_mode = Hit_Mode::last_insert;

      // td.dim = 4;
      // std::cout << "Capacity = 64 * 1024 * 1024 \t"
      //           << "Batch = 1024 * 1024 \t"
      //           << "Key = 64 bits \t"
      //           << "Value = 16 B\n";
      // test_by_discription(td);
      td.dim = 128;
      std::cout << "Capacity = 64 * 1024 * 1024 \t"
                << "Batch = 1024 * 1024 \t"
                << "Key = 64 bits \t"
                << "Value = 512 B\n";
      test_by_discription(td);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}