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
      (offset + start_idx) & (bucket_max_size - 1);
  
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

///TODO: get digests, scores address according keys address
// Prefetch aggressively.
// When bucket size = 16 * X.
// When load factor is high.
template <
  typename K = uint64_t, 
  typename V = float, 
  typename S = uint64_t,
  typename VecV = float4,
  uint32_t GROUP_SIZE = 32, // related to dim
  uint32_t THREAD_BUF = 64> // related to architecture
__global__ void lookup_kernel_with_io_filter(
  Bucket<K, V, S>* buckets, const int bucket_capacity, const uint64_t buckets_num, 
  const int dim, const K* __restrict keys, VecV* __restrict values, 
  S* __restrict scores, bool* __restrict founds, const int n) {
  
  // Digest.
  using D = uint8_t;
  // Vectorized digest.
  using VecD = uint4;
  using BKT = Bucket<K, V, S>;

  constexpr uint32_t BLOCK_SIZE = 128;
  constexpr uint32_t SECTOR_MASK = 0xffffffe0;
  constexpr uint32_t VECTOR_MASK = 0xfffffff0;
  constexpr uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr uint32_t GROUP_BUF = GROUP_SIZE * THREAD_BUF;

  constexpr uint32_t SIZE_VecD = sizeof(VecD) / sizeof(D);
  constexpr uint32_t BUF_D = THREAD_BUF / sizeof(D);
  constexpr uint32_t BUF_VecD = THREAD_BUF / SIZE_VecD;

  // Double buffer.
  constexpr uint32_t SIZE_VecV = sizeof(VecV) / sizeof(V);
  constexpr uint32_t BUF_V = GROUP_BUF / sizeof(V) / 2;
  constexpr uint32_t BUF_VecV = GROUP_BUF / sizeof(VecV) / 2;

  __shared__ VecV sm_values[GROUP_NUM][2][BUF_VecV];
  // Reuse.
  VecD* sm_probing_digests = reinterpret_cast<VecD*>(sm_values);
  // __shared__ VecD sm_probing_digests[BLOCK_SIZE * BUF_VecD];
  // __shared__ int sm_positions[BLOCK_SIZE];

  // Initialization.
  K key;
  S score;
  D digest;
  uint32_t start_idx;
  K* bucket_keys;
  VecV* bucket_values;
  int position = -1;
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      digest = digest_from_hashed<K, D>(hashed_key);
      uint64_t global_idx = static_cast<uint64_t>(
          hashed_key % (buckets_num * bucket_capacity));
      uint64_t bkt_idx = global_idx / bucket_capacity;
      start_idx = global_idx & (bucket_capacity - 1);
      // Probing from the front of the sector.
      // start_idx &= SECTOR_MASK;
      start_idx &= VECTOR_MASK;
      BKT* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys();
      bucket_values = reinterpret_cast<VecV*>(bucket->values());
    } else {
      position = -2;
    }
  }
  
  // Probing.
  // Load partial digests of the bucket.
  uint32_t groupID_VecD = tx / BUF_VecD;
  auto group_VecD = cg::tiled_partition<BUF_VecD>(cg::this_thread_block());
  uint32_t kv_idx_base = blockIdx.x * blockDim.x + groupID_VecD * BUF_VecD;
  if (kv_idx_base < n) {
    uint32_t loop_num =
      (n - kv_idx_base) < BUF_VecD ? (n - kv_idx_base) : BUF_VecD;
    uint32_t rank = group_VecD.thread_rank();
    for (int i = 0; i < loop_num; i++) {
      if (group_VecD.shfl(position, i) == -2) continue;
      uint32_t kv_idx_block = groupID_VecD * BUF_VecD + i;
      VecD* dst = sm_probing_digests + kv_idx_block * BUF_VecD;
      K* bucket_keys_cur = group_VecD.shfl(bucket_keys, i);
      uint32_t start_idx_cur = group_VecD.shfl(start_idx, i);
      uint32_t rank_offset = 
                (start_idx_cur + rank * SIZE_VecD) & (bucket_capacity - 1);
      D* bucket_digests = BKT::digests(bucket_keys_cur, bucket_capacity);
      __pipeline_memcpy_async(dst + rank, bucket_digests + rank_offset, sizeof(VecD));
    }
  }
  __pipeline_commit();
  VecD* sm_digests_local = sm_probing_digests + tx * BUF_VecD;
  if (kv_idx < n && position != -2) {
    __pipeline_wait_prior(0);
    const uint32_t num_VecD = bucket_capacity / SIZE_VecD;
    for (int i = 0; i < num_VecD; i ++) {
      if (i >= BUF_VecD) {
        __pipeline_wait_prior(BUF_VecD - 1);
      }
      VecD digests_vec = sm_digests_local[i % BUF_VecD];
      uint32_t digests[4] = 
                {digests_vec.x, digests_vec.y, digests_vec.z, digests_vec.w};
      // Load the remaining digests of the bucket.
      if (i < num_VecD - BUF_VecD) {
        uint32_t offset_digests = (start_idx + BUF_D +  i * SIZE_VecD) 
                              & (bucket_capacity - 1);
        D* bucket_digests = BKT::digests(bucket_keys, bucket_capacity);
        __pipeline_memcpy_async(sm_digests_local + (i % BUF_VecD),
                                bucket_digests + offset_digests, sizeof(VecD));
      }
      __pipeline_commit();
      int base_position = (start_idx + i * SIZE_VecD) & (bucket_capacity - 1);
      for (int j = 0; j < 4; j ++) {
        uint32_t probing_digests = digests[j];
        for (int k = 0; k < 4; k ++) {
          // Little endian.
          D probing_digest = static_cast<D>(probing_digests);
          probing_digests >>= 8;
          int possible_position = base_position + j * 4 + k;
          if (probing_digest == digest) {
            K possible_key = bucket_keys[possible_position];
            S* bucket_scores = BKT::scores(bucket_keys, bucket_capacity);
            score = bucket_scores[possible_position];
            if (possible_key == key) {
              position = possible_position;
              scores[kv_idx] = score;
              founds[kv_idx] = true;
              goto COPY_VALUE;
            }
          }
          ///TODO:check empty
        }
      }
    }
  }
COPY_VALUE:
  // __shared__ VecV sm_values[GROUP_NUM][2][BUF_VecV];
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  g.sync();
  uint32_t rank = g.thread_rank();
  uint32_t groupID = tx / GROUP_SIZE;
  uint32_t idx_block = groupID * GROUP_SIZE;

  uint32_t copy_mask = g.ballot(position >= 0);
  int rank_cur = __ffs(copy_mask) - 1;
  if (rank_cur >= 0) {
    int pos = g.shfl(position, rank_cur);
    VecV* src = g.shfl(bucket_values, rank_cur);
    __pipeline_memcpy_async(
      sm_values[groupID][0] + rank,
      src + pos * dim + rank, 
      sizeof(VecV)
    );
  }
  __pipeline_commit();
  int i = -1;   // i is used to select buffer.
  while (rank_cur >= 0) {
    i += 1;
    copy_mask &= (copy_mask - 1);
    int rank_next = __ffs(copy_mask) - 1;
    if (rank_next >= 0) {
      int pos = g.shfl(position, rank_next);
      if (pos >= 0) {
        VecV* src = g.shfl(bucket_values, rank_next);
        __pipeline_memcpy_async(
          sm_values[groupID][diff_buf(i)] + rank,
          src + pos * dim + rank, 
          sizeof(VecV)
        );
      }
    }
    __pipeline_commit();
    int pos = g.shfl(position, rank_cur);
    __pipeline_wait_prior(1);
    if (pos >= 0) {
      VecV* src = sm_values[groupID][same_buf(i)];
      uint32_t idx_global = blockIdx.x * blockDim.x + idx_block + rank_cur;
      VecV* dst = values + idx_global * dim;
      dst[rank] = src[rank];
    }
    rank_cur = rank_next;
  }
}
