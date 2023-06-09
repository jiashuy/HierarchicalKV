#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;
//kernel_1 find address, kernel_2 copy value
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v2_kernel1(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n) {
    
  constexpr int BUCKET_SIZE = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);
  size_t global_idx = hashed_key % (buckets_num * BUCKET_SIZE);
  size_t bkt_idx = global_idx / BUCKET_SIZE;
  size_t start_idx = global_idx % BUCKET_SIZE;
  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  S* scores_addr = reinterpret_cast<S*>(keys_addr + BUCKET_SIZE);
  V* values_addr_bucket = bucket->vectors;

  int key_pos = -1;
  uint32_t tile_offset = 0;

  for (tile_offset = 0; tile_offset < BUCKET_SIZE; tile_offset += 1) {

    key_pos =
        (start_idx + tile_offset) & (BUCKET_SIZE - 1);
    K current_key = keys_addr[key_pos];

    if (find_key == current_key) {
      values_addr[key_idx] = values_addr_bucket + key_pos * dim;
      founds[key_idx] = true;
      if (scores != nullptr) {
        scores[key_idx] = scores_addr[key_pos];
      }
      return;
    } else if (current_key == static_cast<K>(EMPTY_KEY)) {
      founds[key_idx] = false;
      return;
    }
  }
}

// A GROUP copy a value
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int rank = tid & (GROUP_SIZE - 1);
  int key_idx = tid >> MASK_WIDTH;
  if (key_idx >= n) return;

  auto v_src = values_addr[key_idx];
  auto v_dst = values + key_idx * dim;
  if (founds[key_idx]) {
    FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A GROUP copy GROUP_SIZE values
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2_test1(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {
  __shared__ V* sm_values_addr[128];
  __shared__ int sm_founds[128];

  int groupID = threadIdx.x >> MASK_WIDTH;
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;

  int loop_num = n - key_idx_base < GROUP_SIZE ? n - key_idx_base : GROUP_SIZE;
  int rank = threadIdx.x & (GROUP_SIZE - 1);
  if (rank < loop_num) {
    sm_values_addr[groupID * GROUP_SIZE + rank] = values_addr[key_idx_base + rank];
    sm_founds[groupID * GROUP_SIZE + rank] = static_cast<int>(founds[key_idx_base + rank]);
  }
  for (int i = 0; i < loop_num; i++) {
    auto v_src = sm_values_addr[groupID * GROUP_SIZE + i];
    auto v_dst = values + (key_idx_base + i) * dim;
    if (sm_founds[groupID * GROUP_SIZE + i])
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A GROUP copy GROUP_SIZE values: prefetch
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2_test2(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {
  __shared__ V* sm_values_addr[128];
  __shared__ int sm_founds[128];
  __shared__ V sm_values[2][128 * 4];

  int groupID = threadIdx.x >> MASK_WIDTH;
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;

  int loop_num = n - key_idx_base < GROUP_SIZE ? n - key_idx_base : GROUP_SIZE;
  int rank = threadIdx.x & (GROUP_SIZE - 1);
  if (rank < loop_num) {
    sm_values_addr[groupID * GROUP_SIZE + rank] = values_addr[key_idx_base + rank];
    sm_founds[groupID * GROUP_SIZE + rank] = static_cast<int>(founds[key_idx_base + rank]);
  }
  for (int i = 0; i < loop_num; i++) {
    auto v_src = sm_values_addr[groupID * GROUP_SIZE + i];
    auto v_dst = values + (key_idx_base + i) * dim;
    if (sm_founds[groupID * GROUP_SIZE + i])
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}