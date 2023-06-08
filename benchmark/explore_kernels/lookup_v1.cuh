#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;
/// one thread deal with one KV pair, including copy value
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v1(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
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
  V* values_addr = bucket->vectors;

  int key_pos = -1;
  uint32_t tile_offset = 0;

  for (tile_offset = 0; tile_offset < BUCKET_SIZE; tile_offset += 1) {

    key_pos =
        (start_idx + tile_offset) & (BUCKET_SIZE - 1);
    K current_key = keys_addr[key_pos];

    if (find_key == current_key) {
      for (int i = 0; i < dim; i += 4) {
        FETCH_FLOAT4(values[key_idx * dim + i]) = 
          FETCH_FLOAT4(values_addr[key_pos * dim + i]);
      }
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