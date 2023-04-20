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
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
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

const float EPSILON = 0.001f;

using namespace nv::merlin;

using namespace benchmark;

enum class Test_Mode {
  pure_hbm = 0,
  hybrid = 1,
};

std::string rep(int n) { return std::string(n, ' '); }

float test_one_api(const API_Select api, const size_t dim,
                   const size_t init_capacity = 64 * 1024 * 1024UL,
                   const size_t key_num_per_op = 1 * 1024 * 1024UL,
                   const size_t hbm4values = 16, const float load_factor = 1.0f,
                   const float hitrate = 0.6f, const bool io_by_cpu = false) {
  using K = uint64_t;
  using M = uint64_t;
  using V = float;
  using Table = nv::merlin::HashTable<K, float, M>;
  using TableOptions = nv::merlin::HashTableOptions;

  size_t free, total;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  if (free / (1 << 30) < hbm4values) {
    return 0.0f;
  }

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = nv::merlin::GB(hbm4values);
  options.io_by_cpu = io_by_cpu;
  options.evict_strategy = EvictStrategy::kCustomized;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

  CUDA_CHECK(
      cudaMemset(h_vectors, 0, key_num_per_op * sizeof(V) * options.dim));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  K* d_keys_out;

  K* d_evict_keys;
  M* d_evict_metas;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, key_num_per_op * sizeof(K)));

  CUDA_CHECK(cudaMalloc(&d_evict_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_metas, key_num_per_op * sizeof(M)));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemset(d_def_val, 2, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, key_num_per_op * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // initialize insert
  // step 1, no need to load load_factor
  uint64_t key_num_init = static_cast<uint64_t>(init_capacity * load_factor);
  const float target_load_factor = key_num_init * 1.0f / init_capacity;
  uint64_t key_num_remain = key_num_init % key_num_per_op == 0
                                ? key_num_per_op
                                : key_num_init % key_num_per_op;
  int32_t loop_num_init = (key_num_init + key_num_per_op - 1) / key_num_per_op;

  K start = 0UL;
  for (int i = 0; i < loop_num_init; i++) {
    uint64_t key_num_cur_insert =
        i == loop_num_init - 1 ? key_num_remain : key_num_per_op;
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_cur_insert, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_cur_insert * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_cur_insert * sizeof(M),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(key_num_cur_insert, d_keys, d_vectors, d_metas,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_cur_insert;
  }
  // step 2
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * init_capacity);
    if (key_num_append <= 0) break;
    key_num_append =
        std::min(static_cast<int64_t>(key_num_per_op), key_num_append);
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_append, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_append * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_append * sizeof(M),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign(key_num_append, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_append;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  create_keys_for_hitrate<K, M>(h_keys, h_metas, key_num_per_op, hitrate,
                                Hit_Mode::last_insert, start, true /*reset*/);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                        cudaMemcpyHostToDevice));
  auto timer = benchmark::Timer<double>();
  switch (api) {
    case API_Select::find: {
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors, d_found, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(key_num_per_op, d_keys, d_vectors, d_metas,
                              stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::assign: {
      timer.start();
      table->assign(key_num_per_op, d_keys, d_def_val, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_metas,
                              d_evict_keys, d_def_val, d_evict_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      break;
    }
    default: {
      std::cout << "[Unsupport API!]\n";
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_metas));

  CudaCheckError();

  float througput =
      key_num_per_op / timer.getResult() / (1024 * 1024 * 1024.0f);
  return througput;
}

static Test_Mode test_mode = Test_Mode::pure_hbm;

void print_title() {
  cout << endl
       << "| dim "
       << "| Capacity<br>(M-KVs) "
       << "| HBM/HMEM<br>(GB) "
       << "| load_factor "
       << "| insert or<br>assign "
       << "|   find "
       << "| find or<br>insert "
       << "| assign ";
  if (Test_Mode::pure_hbm == test_mode) {
    cout << "| insert and<br>evict ";
  }
  cout <<"|\n";

  cout << "|----:"
       //<< "| Capacity<br>(M-KVs) "
       << "|:-------------------:"
       //<< "| HBM/HMEM<br>(GB) "
       << "|:----------------:"
       //<< "| load_factor "
       << "|------------:"
       //<< "| insert or<br>assign "
       << "|:-------------------:"
       //<< "|  find "
       << "|-------:"
       //<< "| find or<br>insert "
       << "|:-----------------:"
       //<< "| assign "
       << "|-------:";
  if (Test_Mode::pure_hbm == test_mode) {
    //<< "| insert and<br>evict "
    cout << "|:-------------------:";
  }
  cout <<"|\n";

}

void test_main(const size_t dim,
               const size_t init_capacity = 64 * 1024 * 1024UL,
               const size_t key_num_per_op = 1 * 1024 * 1024UL,
               const size_t hbm4values = 16, const float load_factor = 1.0f) {
  using V = float;
  int32_t capacity = static_cast<int32_t>(init_capacity / (1024 * 1024));
  size_t hmem4values = init_capacity * dim * sizeof(V) / (1024 * 1024 * 1024);
  hmem4values = hmem4values < hbm4values ? 0 : (hmem4values - hbm4values);

  std::cout << "|" << rep(1) << setw(3) << setfill(' ') << dim << " "
            << "|" << rep(15) << setw(5) << setfill(' ') << capacity << " "
            << "|" << rep(9) << setw(3) << setfill(' ') << hbm4values << " /"
            << setw(3) << setfill(' ') << hmem4values << " "
            << "|" << rep(8) << fixed << setprecision(2) << load_factor << " ";
  std::vector<API_Select> apis{
      API_Select::insert_or_assign,
      API_Select::find,
      API_Select::find_or_insert,
      API_Select::assign,
  };
  if (Test_Mode::pure_hbm == test_mode) {
    apis.push_back(API_Select::insert_and_evict);
  }
  for (auto api : apis) {
    auto res = test_one_api(api, dim, init_capacity, key_num_per_op, hbm4values,
                            load_factor);

    std::cout << "|";
    switch (api) {
      case API_Select::find: {
        std::cout << rep(2);
        break;
      }
      case API_Select::insert_or_assign: {
        std::cout << rep(15);
        break;
      }
      case API_Select::find_or_insert: {
        std::cout << rep(13);
        break;
      }
      case API_Select::assign: {
        std::cout << rep(2);
        break;
      }
      case API_Select::insert_and_evict: {
        std::cout << rep(15);
        break;
      }
      default: {
        std::cout << "[Unsupport API!]";
      }
    }
    std::cout << fixed << setprecision(3) << res << " ";
  }
  std::cout << "|\n";
}

int main() {
  size_t key_num_per_op = 1 * 1024 * 1024UL;
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  cout << endl
       << "## Benchmark" << endl
       << endl
       << "* GPU: 1 x " << props.name << ": " << props.major << "."
       << props.minor << endl
       << "* Key Type = uint64_t" << endl
       << "* Value Type = float32 * {dim}" << endl
       << "* Key-Values per OP = " << key_num_per_op << endl
       << "* ***Throughput Unit: Billion-KV/second***" << endl
       << endl
       << "### On pure HBM mode: " << endl;
  print_title();
  try {
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 0.50f);
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 0.75f);
    test_main(4, 64 * 1024 * 1024UL, key_num_per_op, 32, 1.00f);

    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(16, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(64, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 0.50f);
    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 0.75f);
    test_main(128, 128 * 1024 * 1024UL, key_num_per_op, 64, 1.00f);
    cout << endl;

    cout << "### On HBM+HMEM hybrid mode: " << endl;
    test_mode = Test_Mode::hybrid;
    print_title();
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(64, 128 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 0.50f);
    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 0.75f);
    test_main(64, 1024 * 1024 * 1024UL, key_num_per_op, 56, 1.00f);

    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.50f);
    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 0.75f);
    test_main(128, 64 * 1024 * 1024UL, key_num_per_op, 16, 1.00f);

    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 0.50f);
    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 0.75f);
    test_main(128, 512 * 1024 * 1024UL, key_num_per_op, 56, 1.00f);
    cout << endl;

    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}