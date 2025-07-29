/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <stdio.h>
#include <array>
#include <map>
#include <unordered_map>
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"
#include <cmath>
#include <iostream>

constexpr size_t dim = 28;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;
using V = f32;

template <typename T, int TILE_SIZE>
__global__ void tile_duplicate_kernel(int64_t batch, T* input) {
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr int WARP_SIZE = 32;
  constexpr int KEY_NUM_PER_WARP = WARP_SIZE / TILE_SIZE;
  if (tid < batch) {
    if (tid % KEY_NUM_PER_WARP != 0 and tid % KEY_NUM_PER_WARP != KEY_NUM_PER_WARP - 1) {
      input[tid] = input[tid / KEY_NUM_PER_WARP * KEY_NUM_PER_WARP];
    }
  }
}

void test_assign_score_dead_lock(int64_t bucket_capacity, int64_t max_device_gb, int64_t init_total_gb, 
  float max_load_factor, int64_t batch_size) {

  int64_t min_slice_size = bucket_capacity * sizeof(V) * dim;
  int64_t slice_size = 16UL * 1024 * 1024;
  if (max_device_gb >= 128) {
    slice_size = 16UL * 1024 * 1024 * 1024;
  } else if (max_device_gb >= 16) {
    slice_size = 2UL * 1024 * 1024 * 1024;
  } else if (max_device_gb >= 2) {
    slice_size = 128UL * 1024 * 1024;
  }
  slice_size = std::max(min_slice_size, slice_size);

  int64_t max_capacity = (nv::merlin::GB(max_device_gb) - slice_size) / (dim * sizeof(V));
  int64_t init_capacity = init_total_gb * 1024 * 1024 * 1024 / (sizeof(V) * dim);
  if (init_capacity > max_capacity) {
    init_capacity = max_capacity;
  }

  std::cout << max_capacity << " " << init_capacity << " " << slice_size << "\n"; 

  TableOptions opt;
  // table setting
  opt.max_capacity = max_capacity;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = nv::merlin::GB(max_device_gb);
  opt.max_load_factor = max_load_factor;
  opt.dim = dim;
  opt.max_bucket_size = bucket_capacity;

  using Table =
      nv::merlin::HashTable<i64, V, u64, EvictStrategy::kCustomized>;
  

  // step1
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);
  std::cout << "Fast mode:" << table->is_fast_mode() << "\n";

  // step2

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  test_util::KVMSBuffer<i64, V, u64> buffer;
  buffer.Reserve(batch_size, dim, stream);

  i64 start = 0;
  for (int i = 0; i < max_capacity / batch_size; i++) {
    buffer.ToRange(start, 1, stream);
    start += batch_size;
    buffer.Setscore((u64)i, stream);
    table->insert_or_assign(batch_size, buffer.keys_ptr(), buffer.values_ptr(),
                            buffer.scores_ptr(), stream);

    table->assign_scores(batch_size, buffer.keys_ptr(), buffer.scores_ptr(), stream, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "No duplicated keys, iteration " << i << "\n";
  }

  table->clear(stream);
  start = 0;
  for (int i = 0; i < max_capacity / batch_size; i++) {
    buffer.ToRange(start, 1, stream);
    start += batch_size;
    buffer.Setscore((u64)i, stream);
    table->insert_or_assign(batch_size, buffer.keys_ptr(), buffer.values_ptr(),
                            buffer.scores_ptr(), stream);
    tile_duplicate_kernel<i64, 4><<<(batch_size + 127) / 128, 128, 0, stream>>>(batch_size, buffer.keys_ptr());
    buffer.SyncData(false, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int j = 0; j < 20; j ++) {
      std::cout << buffer.keys_ptr(false)[j] << " ";
    }
    std::cout << "\n";
    table->assign_scores(batch_size, buffer.keys_ptr(), buffer.scores_ptr(), stream, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Duplicated keys, iteration " << i << "\n";
  }
}


__global__ void warp_atomic_cas_kernel(
  int batch, int64_t * keys, int64_t* counts
) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = tid / warpSize;
  int64_t LOCKED_KEY = UINT64_C(0xFFFFFFFFFFFFFFFD);
  int64_t EMPTY_KEY = 0;
  if (warp_id < batch) {
    bool res = false;
    while (not res) {
      int64_t expected_key = EMPTY_KEY;
      int64_t ret = atomicCAS(keys + warp_id, expected_key, LOCKED_KEY);
      if (ret == LOCKED_KEY) break;
    }
    __syncwarp();
    atomicAdd(counts + warp_id, 1);
    atomicCAS(keys + warp_id, LOCKED_KEY, EMPTY_KEY);
  }
}

void test_warp_atomic_cas() {

  int64_t *d_keys, *d_counts;
  constexpr int batch = 32;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(int64_t) * batch));
  CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int64_t) * batch));
  int64_t h_keys[batch], h_counts[batch];

  CUDA_CHECK(cudaMemset(d_keys, 0, sizeof(int64_t) * batch));
  CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int64_t) * batch));

  warp_atomic_cas_kernel<<<batch, 32>>>(batch, d_keys, d_counts);

  CUDA_CHECK(cudaMemcpy(h_keys, d_keys, sizeof(int64_t) * batch, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_counts, d_counts, sizeof(int64_t) * batch, cudaMemcpyDeviceToHost));

  for (int i = 0; i < batch; i++) {
    std::cout << h_keys[i] << " ";
  }
  std::cout << "\n";

  for (int i = 0; i < batch; i++) {
    std::cout << h_counts[i] << " ";
  }
  std::cout << "\n";

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_counts));
}

TEST(DEAD_LOCK_TEST, test_assign_score_dead_lock) { 
  // test_assign_score_dead_lock(128, 30, 90, 0.9, 1024 * 1024UL);
  test_warp_atomic_cas();
}