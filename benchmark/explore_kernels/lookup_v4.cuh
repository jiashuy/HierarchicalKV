#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;

/*
 one thread deal with one key
 a group copy a value cooperatively
*/

// probing using uint4
template <
  class K = uint64_t,
  class V = float,
  class S = uint64_t,
  int GROUP_SIZE = 16,
  class T = uint8_t,
  class Unit = float4,
  int UnitSize = sizeof(Unit) / sizeof(V)>
__global__ void  lookup_kernel_with_io_v4_1(
  Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
  const K* __restrict keys, V* __restrict values, S* __restrict scores,
  bool* __restrict founds, const size_t n) {

  constexpr int BLOCK_SIZE = 128;
  // constexpr int GROUP_SIZE = 32;
  constexpr int bucket_max_size = 128;

  __shared__ int sm_pos      [BLOCK_SIZE];
  __shared__ V* sm_values_ptr[BLOCK_SIZE];
  __shared__ S sm_scores     [BLOCK_SIZE];

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int tx = threadIdx.x;
  sm_pos[tx] = -1;

  int key_idx = blockIdx.x * blockDim.x + tx;
  const K find_key = keys[key_idx];

  K hashed_key = Murmur3HashDevice(find_key);
  uint32_t target_tag = static_cast<uint8_t>(hashed_key >> 32);
  uint32_t target_tags = __byte_perm(target_tag, target_tag, 0x0000);

  int global_idx = hashed_key % (buckets_num * bucket_max_size);
  int bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx & (bucket_max_size - 1);
  start_idx -= start_idx % 16;
  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* key_base = (K*)(bucket->keys_);
  sm_values_ptr[tx] = (V*)(bucket->vectors);

  auto meta_base = reinterpret_cast<S*>(key_base + bucket_max_size);
  auto tag_base = reinterpret_cast<uint4*>((uint8_t*)(key_base) - bucket_max_size);

  for (int offset = 0; offset < bucket_max_size; offset += 16) {
    int key_pos = 
      (offset) & (bucket_max_size - 1);
  
    uint4 tags_ = tag_base[key_pos >> 4];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      int find_result = __vcmpeq4(probe_tags, target_tags);
      bool find_flag = false;
      for (int j = 0; j < 4; j ++) {
        if ((find_result & 0x01) != 0) {
          int possible_pos = key_pos + i * 4 + j;
          K possible_key = key_base[possible_pos];
          if (find_key == possible_key) {
            founds[key_idx] = true;
            sm_pos[tx] = possible_pos;
            sm_scores[tx] = meta_base[possible_pos];
            goto COPY_VALUE;
          }
        }
        find_result >>= 8;
      }
    }
  }
COPY_VALUE:
  g.sync();
  int warpID = tx / GROUP_SIZE;
  for (int i = 0; i < GROUP_SIZE; i++) {
    int idx_block = warpID * GROUP_SIZE + i;
    int pos = sm_pos[idx_block];
    if (pos >= 0) {
      V* v_src = sm_values_ptr[idx_block] + pos * dim;
      int idx_global = (blockIdx.x * blockDim.x) + idx_block;
      scores[idx_global] = sm_scores[idx_block];
      V* v_dst = values + idx_global * dim;
      int rank = g.thread_rank();
      for (int j = rank * UnitSize; j < dim; j += GROUP_SIZE * UnitSize) {
        FETCH_FLOAT4(v_dst[j]) = FETCH_FLOAT4(v_src[j]);
      }
    }
  }
} // end function


// probing using uint4
// prefetch when probing and copy value
template <
  class K = uint64_t,
  class V = float,
  class S = uint64_t,
  int GROUP_SIZE = 16,
  int DIM_BUF = 4, // =4, when dim = 4; =64, when dim = 64; =128 when dim = 128
  class T = uint8_t,
  class Unit = float4,
  int UnitSize = sizeof(Unit) / sizeof(V)>
__global__ void  lookup_kernel_with_io_v4_2(
  Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
  const K* __restrict keys, V* __restrict values, S* __restrict scores,
  bool* __restrict founds, const size_t n) {

  constexpr int BLOCK_SIZE = 128;
  constexpr int bucket_max_size = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;

  __shared__ int sm_pos       [BLOCK_SIZE];
  __shared__ V* sm_values_ptr [BLOCK_SIZE];
  __shared__ S sm_scores      [BLOCK_SIZE];
  __shared__ V sm_values[2][GROUP_NUM][DIM_BUF];
  // __shared__ uint32_t sm_probing_tags[2][BLOCK_SIZE * 4];
  uint32_t* sm_probing_tags = reinterpret_cast<uint32_t*>(sm_values);

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int tx = threadIdx.x;
  sm_pos[tx] = -1;

  int key_idx = (blockIdx.x * blockDim.x) + tx;
  const K find_key = keys[key_idx];

  K hashed_key = Murmur3HashDevice(find_key);
  uint32_t target_tag = static_cast<uint8_t>(hashed_key >> 32);
  uint32_t target_tags = __byte_perm(target_tag, target_tag, 0x0000);

  int global_idx = hashed_key % (buckets_num * bucket_max_size);
  int bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx & (bucket_max_size - 1);
  start_idx -= start_idx % 16;
  start_idx /= 16;
  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* key_base = (K*)(bucket->keys_);
  sm_values_ptr[tx] = (V*)(bucket->vectors);

  auto meta_base = reinterpret_cast<S*>(key_base + bucket_max_size);
  auto tag_base = reinterpret_cast<uint4*>((uint8_t*)(key_base) - bucket_max_size);

  __pipeline_memcpy_async(
    sm_probing_tags + tx * 4,
    tag_base + start_idx, sizeof(uint4)
  );
  __pipeline_commit();
  int loop_num = bucket_max_size / 16;
  for (int offset = 0; offset < loop_num; offset += 1) {
    if ((offset + 1) < loop_num) {
      int offset_bucket = (start_idx + offset + 1) & (loop_num - 1);
      __pipeline_memcpy_async(
        sm_probing_tags + diff_buf(offset) * BLOCK_SIZE * 4  + tx * 4,
        tag_base + offset_bucket, sizeof(uint4)
      );
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    uint4 tags_ = ((uint4*)(sm_probing_tags + same_buf(offset) * BLOCK_SIZE * 4))[tx];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      int find_result = __vcmpeq4(probe_tags, target_tags);
      for (int j = 0; j < 4; j ++) {
        if ((find_result & 0x01) != 0) {
          int tmp = i * 4 + j;
          int possible_pos = ((start_idx + offset) & (loop_num - 1)) * 16 + tmp;
          K possible_key = key_base[possible_pos];
          if (find_key == possible_key) {
            founds[key_idx] = true;
            sm_pos[tx] = possible_pos;
            sm_scores[tx] = meta_base[possible_pos];
            goto COPY_VALUE;
          }
        }
        find_result >>= 8;
      }
    }
  }
COPY_VALUE:
  g.sync();
  int rank = g.thread_rank();
  int groupID = tx / GROUP_SIZE;
  int idx_block = groupID * GROUP_SIZE;
  int pos = sm_pos[idx_block];
  if (pos >= 0) {
    __pipeline_memcpy_async(
      sm_values[0][groupID] + rank * 4,
      sm_values_ptr[idx_block] + pos * dim + rank * 4, 
      sizeof(float4)
    );
  }
  __pipeline_commit();
  for (int i = 0; i < GROUP_SIZE; i++) {
    if ((i + 1) < GROUP_SIZE) {
      int pos = sm_pos[idx_block + i + 1];
      if (pos >= 0) {
        __pipeline_memcpy_async(
          sm_values[diff_buf(i)][groupID] + rank * 4,
          sm_values_ptr[idx_block + i + 1] + pos * dim + rank * 4, 
          sizeof(float4)
        );
      }
    }
    __pipeline_commit();
    int pos = sm_pos[idx_block + i];
    __pipeline_wait_prior(1);
    if (pos >= 0) {
      V* v_src = sm_values[same_buf(i)][groupID];
      int idx_global = blockIdx.x * blockDim.x + idx_block + i;
      scores[idx_global] = sm_scores[idx_block + i];
      V* v_dst = values + idx_global * dim;
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
    }
  }
} // end function
