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

  __shared__ V sm_values[GROUP_NUM][BUF_NUM][VALUE_PER_BUF][BUF_V];

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
  for (int i = 0; i < GROUP_SIZE; i ++) {
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur != OccupyResult::CONTINUE) continue;
    
    K expected_key = static_cast<K>(EMPTY_KEY);

    AtomicKey<K>* current_key;
    AtomicScore<S>* current_score;

    K local_min_score_key = static_cast<K>(EMPTY_KEY);

    S local_min_score_val = MAX_SCORE;
    S temp_min_score_val = MAX_SCORE;
    S global_min_score_val;
    int local_min_score_pos = -1;
    int key_pos_local;
    K evicted_key_local;

    unsigned vote = 0;
    bool result = false;
    do {
      expected_key = static_cast<K>(EMPTY_KEY);

      local_min_score_key = static_cast<K>(EMPTY_KEY);

      local_min_score_val = MAX_SCORE;
      temp_min_score_val = MAX_SCORE;
      local_min_score_pos = -1;

      vote = 0;
      result = false;
      auto start_idx_cur = g.shfl(start_idx, i);
      auto bucket_cur = g.shfl(bucket, i);

      for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
          tile_offset += GROUP_SIZE) {
        key_pos_local = (start_idx_cur + tile_offset + g.thread_rank()) % bucket_max_size;

        current_score = bucket_cur->scores(key_pos_local);

        // Step 4: record min score location.
        temp_min_score_val = current_score->load(cuda::std::memory_order_relaxed);
        if (temp_min_score_val < local_min_score_val) {
          expected_key =
              bucket_cur->keys(key_pos_local)->load(cuda::std::memory_order_relaxed);
          if (expected_key != static_cast<K>(LOCKED_KEY) &&
              expected_key != static_cast<K>(EMPTY_KEY)) {
            local_min_score_key = expected_key;
            local_min_score_val = temp_min_score_val;
            local_min_score_pos = key_pos_local;
          }
        }
      }
      // Step 5: insert by evicting some one.
      global_min_score_val =
          cg::reduce(g, local_min_score_val, cg::less<S>());
      auto insert_score_cur = g.shfl(insert_score, i);
      if (insert_score_cur < global_min_score_val) {
        if (g.thread_rank() == i) {
          occupy_result = OccupyResult::REFUSED;
          if (evicted_scores != nullptr && scores != nullptr) {
            evicted_scores[key_idx] = scores[key_idx];
          }
          evicted_keys[key_idx] = keys[key_idx];
        }
        goto FINISH;
      }
      vote = g.ballot(local_min_score_val <= global_min_score_val);
      if (vote) {
        int src_lane_cur = __ffs(vote) - 1;
        result = false;
        if (src_lane_cur == g.thread_rank()) {
          // TBD: Here can be compare_exchange_weak. Do benchmark.
          current_key = bucket_cur->keys(local_min_score_pos);
          current_score = bucket_cur->scores(local_min_score_pos);
          evicted_key_local = local_min_score_key;
          result = current_key->compare_exchange_strong(
              local_min_score_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);

          // Need to recover when fail.
          if (result && (current_score->load(cuda::std::memory_order_relaxed) >
                        global_min_score_val)) {
            current_key->store(local_min_score_key,
                              cuda::std::memory_order_relaxed);
            result = false;
          }
        }
        result = g.shfl(result, src_lane_cur);
        if (result) {
          // Not every `evicted_key` is correct expect the `src_lane` thread.
          key_pos_local = g.shfl(local_min_score_pos, src_lane_cur);
          evicted_key_local = g.shfl(evicted_key_local, src_lane_cur);
          if (g.thread_rank() == i) {
            evicted_key = evicted_key_local;
            key_pos = key_pos_local;
            if (evicted_key == static_cast<K>(RECLAIM_KEY)) {
              atomicAdd(&(buckets_size[bkt_idx]), 1);
              occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
            } else {
              occupy_result = OccupyResult::EVICT;
              evicted_keys[key_idx] = evicted_key;
              if (scores != nullptr) {
                evicted_scores[key_idx] = global_min_score_val;
              }
            }
            update_score(bucket, key_pos, scores, key_idx);
            bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
          }
          goto FINISH;
        } 
      } 
      occupy_result_cur = OccupyResult::CONTINUE;
FINISH:
      occupy_result_cur = g.shfl(occupy_result, i);
    } while (occupy_result_cur == OccupyResult::CONTINUE);
  }

  uint32_t groupID = threadIdx.x / GROUP_SIZE;
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