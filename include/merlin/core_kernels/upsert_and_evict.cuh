/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kernel_utils.cuh"
// #include "/Workspace/HierarchicalKV/benchmark/explore_kernels/insert_and_evict.cuh"

namespace nv {
namespace merlin {
template <class S>
static __forceinline__ __device__ S device_nano() {
  S mclk;
  asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(mclk));
  return mclk;
}

template <
typename K = uint64_t,
typename V = float,
typename S = uint64_t,
typename VecV = float4,
typename D = uint8_t,
typename VecD = uint4,
uint32_t BUCKET_SIZE = 128,
uint32_t BLOCK_SIZE = 128,
uint32_t GROUP_SIZE = 32
uint32_t BUF_V = 64>
__global__ void upsert_and_evict_kernel_pipeline_unique(
  Bucket<K, V, S>* buckets, int32_t* buckets_size, const uint64_t buckets_num, 
  const uint32_t dim, const K* __restrict keys, const V* __restrict values, 
  const S* __restrict scores, K* __restrict evicted_keys, V* __restrict evicted_values,
  S* __restrict evicted_scores, uint32_t n) {
  using BKT = Bucket<K, V, S>;
  constexpr VecD_SIZE = sizeof(VecD) / sizeof(D);
  constexpr uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  __shared__ D sm_bucket_digests[GROUP_NUM][2][BUCKET_SIZE];
  __shared__ S sm_bucket_scores[GROUP_NUM][2][BUCKET_SIZE];
  __shared__ V sm_values_buffer[GROUP_NUM][2][BUF_V];

  // Initialization.
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  uint32_t kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
  K key;
  S score;
  D digest;
  uint64_t bkt_idx = 0;
  uint32_t start_idx = 0;
  K* bucket_keys_ptr {nullptr};
  V* bucket_values_ptr {nullptr};
  uint32_t bucket_size = 0;
  OccupyResult occupy_result{OccupyResult::INITIAL};
  uint32_t key_pos = 0;
  if (kv_idx < n && ) {
    key = keys[kv_idx];
    if (!IS_RESERVED_KEY(insert_key)) {
      score =
        scores != nullptr ? scores[kv_idx] : static_cast<S>(MAX_SCORE);
      const K hashed_key = Murmur3HashDevice(key);
      digest = digest_from_hashed<K>(hashed_key);
      uint64_t global_idx = static_cast<uint64_t>(
          hashed_key % (buckets_num * BUCKET_SIZE));
      bkt_idx = global_idx / BUCKET_SIZE;
      start_idx = global_idx & (BUCKET_SIZE - 1);
      Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys();
      bucket_values_ptr = bucket->values();
      bucket_size = buckets_size[bkt_idx];
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  uint32_t rank = g.thread_rank();
  uint32_t groupID = threadIdx.x / GROUP_SIZE;
  for (int32_t i = 0; i < GROUP_SIZE; i++) {
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur == OccupyResult::INITIAL) {
      D* dst = sm_bucket_digests[groupID][0];
      auto keys_ptr_cur = g.shfl(bucket_keys_ptr, i);
      D* src = BKT::digests(keys_ptr_cur, BUCKET_SIZE);
      if (rank * VecD_SIZE < BUCKET_SIZE)
        __pipeline_memcpy_async(dst + rank * VecD_SIZE, 
                                src + rank * VecD_SIZE, sizeof(VecD));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    if (occupy_result_cur == OccupyResult::INITIAL) {
      D* src = sm_bucket_digests[groupID][0];
      uint32_t digests_ = reinterpret_cast<uint32_t>(src)[rank];
      

    }

  }

}

template <class K, class V, class S, uint32_t GROUP_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core_beta(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores,
    K* __restrict evicted_keys, V* __restrict evicted_values,
    S* __restrict evicted_scores, size_t N) {
  constexpr uint32_t BLOCK_SIZE = 128U;
  constexpr uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr uint32_t BLOCK_BUF = 8192U;
  constexpr uint32_t GROUP_BUF = BLOCK_BUF / GROUP_NUM;

  constexpr uint32_t BUF_NUM = 2;
  constexpr uint32_t VALUE_PER_BUF = 2;
  constexpr uint32_t BUF_V = GROUP_BUF / VALUE_PER_BUF / BUF_NUM / sizeof(V);

  constexpr uint32_t GROUP_BUF_S = GROUP_BUF / sizeof(S);
  constexpr uint32_t BUCKET_S = GROUP_BUF_S / 2;

  __shared__ V sm_values[GROUP_NUM][BUF_NUM][VALUE_PER_BUF][BUF_V];

  S* sm_scores = (S*)sm_values;

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;
  size_t key_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  int key_pos = -1;
  K insert_key;
  S insert_score;
  const V* insert_value = values + key_idx * dim;
  size_t bkt_idx = 0;
  size_t start_idx = 0;
  K evicted_key;
  OccupyResult occupy_result{OccupyResult::INITIAL};
  Bucket<K, V, S>* bucket;


  if (key_idx < N) {
    insert_key = keys[key_idx];
    if (IS_RESERVED_KEY(insert_key)) {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  if (occupy_result == OccupyResult::INITIAL) {
    insert_score =
        scores != nullptr ? scores[key_idx] : static_cast<S>(MAX_SCORE);

    bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    K expected_key = static_cast<K>(EMPTY_KEY);
    AtomicKey<K>* current_key;
    uint8_t digest = get_digest<K>(insert_key);
    uint32_t target_digests = __byte_perm(digest, digest, 0x0000);
    bool result = false;

    for (int tile_offset = 0; tile_offset < bucket_max_size + 16; tile_offset += 16) {
      key_pos = (start_idx - (start_idx % 16) + tile_offset) % bucket_max_size;
      uint8_t* digests_ptr = bucket->digests(key_pos);
      uint4* digests_vec_ptr = (uint4*)digests_ptr;
      uint4 tags_ = digests_vec_ptr[0];
      uint32_t tags[4] = 
                {tags_.x, tags_.y, tags_.z, tags_.w};    
      for (int i = 0; i < 4; i ++) {
        uint32_t probing_digests = tags[i];
        int find_result = __vcmpeq4(probing_digests, target_digests);
        if (find_result != 0) {
          for (int j = 0; j < 4; j ++) {
            if ((find_result & 0x01) != 0) {
              int possible_pos = key_pos + i * 4 + j;
              expected_key = insert_key;
              current_key = bucket->keys(possible_pos);
              result = current_key->compare_exchange_strong(
                  expected_key, static_cast<K>(LOCKED_KEY),
                  cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
              if (result) {
                key_pos = possible_pos;
                update_score(bucket, key_pos, scores, key_idx);
                bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
                occupy_result = OccupyResult::DUPLICATE;
                goto REDUCE;
              }
            }
            find_result >>= 8;
          }
        }
        uint32_t empty_digests = __byte_perm(empty_digest<K>(), empty_digest<K>(), 0x0000);
        find_result = __vcmpeq4(probing_digests, empty_digests);
        if (find_result != 0) {
          for (int j = 0; j < 4; j ++) {
            if ((find_result & 0x01) != 0) {
              int possible_pos = key_pos + i * 4 + j;
              if (tile_offset == 0 && possible_pos < start_idx) continue;
              expected_key = static_cast<K>(EMPTY_KEY);;
              current_key = bucket->keys(possible_pos);
              result = current_key->compare_exchange_strong(
                  expected_key, static_cast<K>(LOCKED_KEY),
                  cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
              if (result) {
                key_pos = possible_pos;
                update_score(bucket, key_pos, scores, key_idx);
                bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
                occupy_result = OccupyResult::OCCUPIED_EMPTY;
                atomicAdd(&(buckets_size[bkt_idx]), 1);
                goto REDUCE;
              }
            }
            find_result >>= 8;
          }
        }              
      }
    }
    occupy_result = OccupyResult::CONTINUE; 
  }
REDUCE:
  uint32_t groupID = threadIdx.x / GROUP_SIZE;
  // Get the KV-pair with min score and evict it.
  uint32_t reduce_mask = g.ballot(occupy_result == OccupyResult::CONTINUE);
  int rank_next = __ffs(reduce_mask) - 1;
  if (rank_next >= 0) {
    S* dst = sm_scores + groupID * GROUP_BUF_S;
    auto bucket_next = g.shfl(bucket, rank_next);
    auto src = bucket_next->scores(0);
    int rank = g.thread_rank();
    __pipeline_memcpy_async(dst + rank * 2, 
                            src + rank * 2, sizeof(uint4));
    __pipeline_memcpy_async(dst + rank * 2 + bucket_max_size / 2,
                            src + rank * 2 + bucket_max_size / 2, sizeof(uint4));
  }
  __pipeline_commit();
  int i = -1;
  int rank = g.thread_rank();
  while (reduce_mask != 0) {
    i += 1;
    int rank_cur = __ffs(reduce_mask) - 1;
    reduce_mask &= (reduce_mask - 1);
    int rank_next = __ffs(reduce_mask) - 1;
    if (rank_next >= 0) {
      S* dst = sm_scores + groupID * GROUP_BUF_S + diff_buf(i) * BUCKET_S;
      auto bucket_next = g.shfl(bucket, rank_next);
      auto src = bucket_next->scores(0);
      __pipeline_memcpy_async(dst + rank * 2, 
                              src + rank * 2, sizeof(uint4));
      __pipeline_memcpy_async(dst + rank * 2 + bucket_max_size / 2,
                              src + rank * 2 + bucket_max_size / 2, sizeof(uint4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    auto  occupy_result_cur = g.shfl(occupy_result, rank_cur);
    S* dst = sm_scores + groupID * GROUP_BUF_S + same_buf(i) * BUCKET_S;
    auto bucket_cur = g.shfl(bucket, rank_cur);
    auto src = bucket_cur->scores(0);
    while (occupy_result_cur == OccupyResult::CONTINUE) {
      int min_pos_local = -1;
      S min_score_local = MAX_SCORE;
      for (int j = rank; j < bucket_max_size; j += GROUP_SIZE) {
        S temp_score = dst[j];
        if (temp_score < min_score_local) {
          min_score_local = temp_score;
          min_pos_local = j;
        }
      }
      const S min_score_global =
          cg::reduce(g, min_score_local, cg::less<S>());

      auto insert_score_cur = g.shfl(insert_score, rank_cur);
      if (insert_score_cur < min_score_global) {
        if (rank == rank_cur) {
          occupy_result = OccupyResult::REFUSED;
          if (evicted_scores != nullptr && scores != nullptr) {
            evicted_scores[key_idx] = insert_score;
          }
          evicted_keys[key_idx] = insert_key;
        }
        break;
      }
      uint32_t vote = g.ballot(min_score_local <= min_score_global);
      if (vote) {
        int src_lane = __ffs(vote) - 1;
        int min_pos_global = g.shfl(min_pos_local, src_lane);
        if (rank == rank_cur) {
          dst[min_pos_global] = MAX_SCORE; // Mark visited.
          auto min_score_key = bucket_cur->keys(min_pos_global);
          auto expected_key = min_score_key->load(cuda::std::memory_order_relaxed);
          if (expected_key == static_cast<K>(LOCKED_KEY) ||
            expected_key == static_cast<K>(EMPTY_KEY)) {
            goto FINISH;
          }
          auto min_score_ptr = bucket_cur->scores(min_pos_global);
          bool result = min_score_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
          if (!result) {
            goto FINISH;
          }
          if (min_score_ptr->load(cuda::std::memory_order_relaxed) >
                    min_score_global) {
            min_score_key->store(expected_key,
                        cuda::std::memory_order_relaxed);
            goto FINISH;
          }
          evicted_key = expected_key;
          key_pos = min_pos_global;
          if (evicted_key == static_cast<K>(RECLAIM_KEY)) {
            atomicAdd(&(buckets_size[bkt_idx]), 1);
            occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
          } else {
            occupy_result = OccupyResult::EVICT;
            evicted_keys[key_idx] = evicted_key;
            if (scores != nullptr) {
              evicted_scores[key_idx] = min_score_global;
            }
          }
          update_score(bucket, key_pos, scores, key_idx);
          bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
        }
FINISH:
        occupy_result_cur = g.shfl(occupy_result, rank_cur);
      }
    }
  }
  auto occupy_result_next = g.shfl(occupy_result, 0);
  if (occupy_result_next != OccupyResult::ILLEGAL) {
    auto insert_value_next = g.shfl(insert_value, 0);
    auto key_idx_next = g.shfl(key_idx, 0);
    auto key_pos_next = g.shfl(key_pos, 0);
    auto bucket_next = g.shfl(bucket, 0);

    const V* src = insert_value_next;
    V* dst = sm_values[groupID][0][0];
    int rank_offset = g.thread_rank() * 4;
    if (rank_offset < dim)
      __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
    if (occupy_result_next == OccupyResult::EVICT) {
      const V* src = bucket_next->vectors + key_pos_next * dim;
      V* dst = sm_values[groupID][0][1];
      int rank_offset = g.thread_rank() * 4;
      if (rank_offset < dim)
        __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));      
    }
  }
  __pipeline_commit();

  for (int i = 0; i < GROUP_SIZE; i++) {
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      if (occupy_result_next != OccupyResult::ILLEGAL) {
        auto insert_value_next = g.shfl(insert_value, i + 1);
        auto key_idx_next = g.shfl(key_idx, i + 1);
        auto key_pos_next = g.shfl(key_pos, i + 1);
        auto bucket_next = g.shfl(bucket, i + 1);

        const V* src = insert_value_next;
        V* dst = sm_values[groupID][diff_buf(i)][0]; 
        int rank_offset = g.thread_rank() * 4;
        if (rank_offset < dim)
          __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
        if (occupy_result_next == OccupyResult::EVICT) {
          const V* src = bucket_next->vectors + key_pos_next * dim;
          V* dst = sm_values[groupID][diff_buf(i)][1];
          int rank_offset = g.thread_rank() * 4;
          if (rank_offset < dim)
            __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
        }
      }
    }
    __pipeline_commit();
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur != OccupyResult::ILLEGAL) {
      auto insert_value_cur = g.shfl(insert_value, i);
      auto key_idx_cur = g.shfl(key_idx, i);
      auto key_pos_cur = g.shfl(key_pos, i);
      auto bucket_cur = g.shfl(bucket, i);

      if (occupy_result_cur == OccupyResult::REFUSED) {
        V* src = sm_values[groupID][same_buf(i)][0];
        V* dst = evicted_values + key_idx_cur * dim;
        int rank_offset = g.thread_rank() * 4;
        __pipeline_wait_prior(1);
        if (rank_offset < dim) {
          float4 vecV = ((float4*)(src + rank_offset))[0];
          ((float4*)(dst + rank_offset))[0] = vecV;
        }
        continue;
      }

      if (occupy_result_cur == OccupyResult::EVICT) {
        V* src = sm_values[groupID][same_buf(i)][1];
        V* dst = evicted_values + key_idx_cur * dim;
        int rank_offset = g.thread_rank() * 4;
        __pipeline_wait_prior(1);
        if (rank_offset < dim) {
          float4 vecV = ((float4*)(src + rank_offset))[0];
          ((float4*)(dst + rank_offset))[0] = vecV;
        }
      }

      V* src = sm_values[groupID][same_buf(i)][0];
      V* dst = bucket_cur->vectors + key_pos_cur * dim;
      int rank_offset = g.thread_rank() * 4;
      __pipeline_wait_prior(1);
      if (rank_offset < dim) {
        float4 vecV = ((float4*)(src + rank_offset))[0];
        ((float4*)(dst + rank_offset))[0] = vecV;
      }

      if (g.thread_rank() == i) {
        (bucket->keys(key_pos))
            ->store(insert_key, cuda::std::memory_order_relaxed);
      }

    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores,
    K* __restrict evicted_keys, V* __restrict evicted_values,
    S* __restrict evicted_scores, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const S insert_score =
        scores != nullptr ? scores[key_idx] : static_cast<S>(MAX_SCORE);
    const V* insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, S>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, S, TILE_SIZE>(
            g, bucket, insert_key, insert_score, evicted_key, start_idx,
            key_pos, src_lane, bucket_max_size);
      }
      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      copy_vector<V, TILE_SIZE>(g, insert_value, evicted_values + key_idx * dim,
                                dim);
      continue;
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == src_lane) {
        evicted_keys[key_idx] = evicted_key;
      }
      if (scores != nullptr) {
        evicted_scores[key_idx] = scores[key_idx];
      }
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                evicted_values + key_idx * dim, dim);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_score(bucket, key_pos, scores, key_idx);
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S>
struct SelectUpsertAndEvictKernelWithIO {
  static void execute_kernel(
      const float& load_factor, const int& block_size,
      const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
      cudaStream_t& stream, const size_t& n,
      const Table<K, V, S>* __restrict table, const K* __restrict keys,
      const V* __restrict values, const S* __restrict scores,
      K* __restrict evicted_keys, V* __restrict evicted_values,
      S* __restrict evicted_scores) {
    if (true) {
      const unsigned int tile_size = 32;
      const size_t N = n;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core_beta<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);
    } else
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);

    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);

    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, scores,
              evicted_keys, evicted_values, evicted_scores, N);
    }
    return;
  }
};

}  // namespace merlin
}  // namespace nv