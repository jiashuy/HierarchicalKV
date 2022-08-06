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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "merlin/initializers.cuh"
#include "merlin/optimizers.cuh"
#include "merlin_hashtable.cuh"

using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
template <class K, class M>
void create_random_keys(K *h_keys, M *h_metas, int key_num_per_op) {
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
    h_metas[i] = getTimestamp();
    i++;
  }
}

std::string rep(int n) { return std::string(n, ' '); }

template <class K, class M>
void create_continuous_keys(K *h_keys, M *h_metas, int key_num_per_op,
                            K start = 0) {
  for (K i = 0; i < key_num_per_op; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <class V, size_t DIM>
struct ValueArray {
  V value[DIM];
};

template <size_t DIM>
void test_main(size_t init_capacity = 64 * 1024 * 1024UL,
               size_t key_num_per_op = 1 * 1024 * 1024UL,
               size_t hbm4values = 16, float load_factor = 1.0) {
  using K = uint64_t;
  using M = uint64_t;
  using Vector = ValueArray<float, DIM>;
  using Table = nv::merlin::HashTable<K, float, M, DIM>;

  size_t free, total;
  cudaSetDevice(0);
  cudaMemGetInfo(&free, &total);

  if (free / (1 << 30) < hbm4values) {
    return;
  }

  K *h_keys;
  M *h_metas;
  Vector *h_vectors;
  bool *h_found;

  std::unique_ptr<Table> table_ =
      std::make_unique<Table>(init_capacity,              /* init_capacity */
                              init_capacity,              /* max_size */
                              nv::merlin::GB(hbm4values), /* hbm4values */
                              0.75,                       /* max_load_factor */
                              128,                        /* buckets_max_size */
                              nullptr,                    /* initializer */
                              true,                       /* primary */
                              1024                        /* block_size */
      );

  cudaMallocHost(&h_keys, key_num_per_op * sizeof(K));          // 8MB
  cudaMallocHost(&h_metas, key_num_per_op * sizeof(M));         // 8MB
  cudaMallocHost(&h_vectors, key_num_per_op * sizeof(Vector));  // 256MB
  cudaMallocHost(&h_found, key_num_per_op * sizeof(bool));      // 4MB

  cudaMemset(h_vectors, 0, key_num_per_op * sizeof(Vector));

  K *d_keys;
  M *d_metas = nullptr;
  Vector *d_vectors;
  Vector *d_def_val;
  Vector **d_vectors_ptr;
  bool *d_found;

  cudaMalloc(&d_keys, key_num_per_op * sizeof(K));                // 8MB
  cudaMalloc(&d_metas, key_num_per_op * sizeof(M));               // 8MB
  cudaMalloc(&d_vectors, key_num_per_op * sizeof(Vector));        // 256MB
  cudaMalloc(&d_def_val, key_num_per_op * sizeof(Vector));        // 256MB
  cudaMalloc(&d_vectors_ptr, key_num_per_op * sizeof(Vector *));  // 8MB
  cudaMalloc(&d_found, key_num_per_op * sizeof(bool));            // 4MB

  cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
             cudaMemcpyHostToDevice);

  cudaMemset(d_vectors, 1, key_num_per_op * sizeof(Vector));
  cudaMemset(d_def_val, 2, key_num_per_op * sizeof(Vector));
  cudaMemset(d_vectors_ptr, 0, key_num_per_op * sizeof(Vector *));
  cudaMemset(d_found, 0, key_num_per_op * sizeof(bool));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  K start = 0UL;
  float cur_load_factor = table_->load_factor();
  auto start_insert_or_assign = std::chrono::steady_clock::now();
  auto end_insert_or_assign = std::chrono::steady_clock::now();
  auto start_find = std::chrono::steady_clock::now();
  auto end_find = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff_insert_or_assign;
  std::chrono::duration<double> diff_find;

  while (cur_load_factor < load_factor) {
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_per_op, start);
    cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
               cudaMemcpyHostToDevice);

    start_insert_or_assign = std::chrono::steady_clock::now();
    table_->insert_or_assign(d_keys, reinterpret_cast<float *>(d_vectors),
                             d_metas, key_num_per_op, false, stream);
    end_insert_or_assign = std::chrono::steady_clock::now();
    diff_insert_or_assign = end_insert_or_assign - start_insert_or_assign;

    start_find = std::chrono::steady_clock::now();
    table_->find(d_keys, reinterpret_cast<float *>(d_vectors), d_found,
                 key_num_per_op, nullptr, stream);
    end_find = std::chrono::steady_clock::now();
    diff_find = end_find - start_find;

    cur_load_factor = table_->load_factor();

    start += key_num_per_op;
  }

  size_t hmem4values =
      init_capacity * DIM * sizeof(float) / (1024 * 1024 * 1024);
  hmem4values = hmem4values < hbm4values ? 0 : (hmem4values - hbm4values);
  float insert_tput =
      key_num_per_op / diff_insert_or_assign.count() / (1024 * 1024 * 1024.0);
  float find_tput = key_num_per_op / diff_find.count() / (1024 * 1024 * 1024.0);

  cout << "|" << rep(1) << setw(3) << setfill(' ') << DIM << " "
       << "|" << rep(1) << setw(11) << setfill(' ') << init_capacity << " "
       << "|" << rep(9) << key_num_per_op << " "
       << "|" << rep(8) << fixed << setprecision(2) << load_factor << " "
       << "|" << rep(5) << setw(3) << setfill(' ') << hbm4values << " "
       << "|" << rep(6) << setw(3) << setfill(' ') << hmem4values << " "
       << "|" << rep(20) << fixed << setprecision(3) << insert_tput << " "
       << "|" << rep(8) << fixed << setprecision(3) << find_tput << " |"
       << endl;

  cudaStreamDestroy(stream);

  cudaFreeHost(h_keys);
  cudaFreeHost(h_metas);
  cudaFreeHost(h_found);

  cudaFree(d_keys);
  cudaFree(d_metas);
  cudaFree(d_vectors);
  cudaFree(d_def_val);
  cudaFree(d_vectors_ptr);
  cudaFree(d_found);

  return;
}

void print_title() {
  cout << endl
       << "| dim "
       << "|    capacity "
       << "| keys_num_per_op "
       << "| load_factor "
       << "| HBM(GB) "
       << "| HMEM(GB) "
       << "| insert_or_assign(G-KV/s) "
       << "| find(G-KV/s) |" << endl;
  cout << "|----:"
       //<< "| capacity "
       << "|------------:"
       //<< "| keys_num_per_op "
       << "|----------------:"
       //<< "| load_factor "
       << "|------------:"
       //<< "| HBM(GB) "
       << "|--------:"
       //<< "| HMEM(GB) "
       << "|---------:"
       //<< "| insert_or_assign(G-KV/s) "
       << "|-------------------------:"
       //<< "| find(G-KV/s) "
       << "|-------------:|" << endl;
}

int main() {
  cout << "On pure HBM mode: " << endl;
  print_title();
  test_main<4>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.50);
  test_main<4>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.75);
  test_main<4>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 1.00);
  test_main<16>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.50);
  test_main<16>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.75);
  test_main<16>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 1.00);

  test_main<64>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.50);
  test_main<64>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.75);
  test_main<64>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 1.00);

  test_main<128>(128 * 1024 * 1024UL, 1024 * 1024UL, 64, 0.50);
  test_main<128>(128 * 1024 * 1024UL, 1024 * 1024UL, 64, 0.75);
  test_main<128>(128 * 1024 * 1024UL, 1024 * 1024UL, 64, 1.00);
  cout << endl;

  cout << "On HBM+HMEM hybrid mode: " << endl;
  print_title();
  test_main<64>(128 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.50);
  test_main<64>(128 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.75);
  test_main<64>(128 * 1024 * 1024UL, 1024 * 1024UL, 16, 1.00);

  test_main<64>(1024 * 1024 * 1024UL, 1024 * 1024UL, 56, 0.50);
  test_main<64>(1024 * 1024 * 1024UL, 1024 * 1024UL, 56, 0.75);
  test_main<64>(1024 * 1024 * 1024UL, 1024 * 1024UL, 56, 1.00);

  test_main<128>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.50);
  test_main<128>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 0.75);
  test_main<128>(64 * 1024 * 1024UL, 1024 * 1024UL, 16, 1.00);

  test_main<128>(512 * 1024 * 1024UL, 1024 * 1024UL, 56, 0.50);
  test_main<128>(512 * 1024 * 1024UL, 1024 * 1024UL, 56, 0.75);
  test_main<128>(512 * 1024 * 1024UL, 1024 * 1024UL, 56, 1.00);
  cout << endl;

  return 0;
}