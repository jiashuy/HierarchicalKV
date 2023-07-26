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

namespace nv {
namespace merlin {

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
struct UpsertAndEvictKernelParams {
  LookupKernelParams(Bucket<K, V, S>* __restrict buckets_, size_t buckets_num_,
                     uint32_t dim_, const K* __restrict keys_,
                     V* __restrict values_, S* __restrict scores_,
                     bool* __restrict founds_, size_t n_)
      : buckets(buckets_),
        buckets_num(buckets_num_),
        dim(dim_),
        keys(keys_),
        values(values_),
        scores(scores_),
        founds(founds_),
        n(n_) {}
  Bucket<K, V, S>* __restrict buckets;
  size_t buckets_num;
  uint32_t dim;
  const K* __restrict keys;
  V* __restrict values;
  S* __restrict scores;
  bool* __restrict founds;
  size_t n;
};

template <
typename K = uint64_t,
typename V = float,
typename S = uint64_t,
typename VecV = float4,
typename D = uint8_t,
typename VecD = uint4,
uint32_t BUCKET_SIZE = 128,
uint32_t BLOCK_SIZE = 64,
uint32_t GROUP_SIZE = 8,
uint32_t BUF_V = 64>
__global__ void upsert_and_evict_kernel_unique_v2(
  Bucket<K, V, S>* buckets, int32_t* buckets_size, const uint64_t buckets_num, 
  const uint32_t dim, const K* __restrict keys, const V* __restrict values, 
  const S* __restrict scores, K* __restrict evicted_keys, V* __restrict evicted_values,
  S* __restrict evicted_scores, uint32_t n, size_t* evict_number) {
  
  using BKT = Bucket<K, V, S>;
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;

  K key {static_cast<K>(EMPTY_KEY)};
  S score {static_cast<S>(EMPTY_SCORE)};
  uint32_t target_digests {0};
  BKT* bucket {nullptr};
  K* bucket_keys_ptr {nullptr};
  uint32_t key_pos = {0};
  uint32_t evict_idx {0};
  int32_t* bucket_size_address {nullptr};
  uint32_t bucket_size {0};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  if (kv_idx < n) {
    key = keys[kv_idx];
    score = scores != nullptr ? scores[kv_idx] : static_cast<S>(MAX_SCORE);
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      D digest = digest_from_hashed<K>(hashed_key);
      target_digests = __byte_perm(digest, digest, 0x0000);
      uint64_t global_idx = static_cast<uint64_t>(
          hashed_key % (buckets_num * BUCKET_SIZE));
      key_pos = global_idx & (BUCKET_SIZE - 1);
      uint64_t bkt_idx = global_idx / BUCKET_SIZE;
      bucket_size_address = buckets_size + bkt_idx;
      bucket = buckets + bkt_idx;
      bucket_size = *bucket_size_address;
      bucket_keys_ptr = bucket->keys();
    } else {
      return;
    }
  } else {
    return;
  }

  constexpr uint32_t LEN_VecD = sizeof(VecD) / sizeof(D);
  constexpr uint32_t MASK_VecD = 0xffffffffU - (LEN_VecD - 1);
  for (int offset = 0; offset < BUCKET_SIZE + LEN_VecD; offset += LEN_VecD) {
    if (occupy_result != OccupyResult::INITIAL) break;
    uint32_t pos_cur = ((key_pos & MASK_VecD) + offset) & (BUCKET_SIZE - 1);
    D* digests_ptr = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, pos_cur);
    VecD digests_vec = *(reinterpret_cast<VecD*>(digests_ptr));
    uint32_t digests_arr[4] = 
              {digests_vec.x, digests_vec.y, digests_vec.z, digests_vec.w}; 
    for (int i = 0; i < 4; i++) {
      uint32_t probing_digests = digests_arr[i];
      uint32_t possible_pos = 0;
      bool result = false;
      int find_result = __vcmpeq4(probing_digests, target_digests);
      find_result &= 0x01010101;
      do {
        if (find_result == 0) break;
        // Little endian.
        uint32_t index = (__ffs(find_result) - 1) >> 3;
        find_result &= (find_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        auto current_key = BKT::keys(bucket_keys_ptr, possible_pos);
        K expected_key = key;
        // __threadfence();
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
        // __threadfence();
      } while (!result);
      if (result) {
        key_pos = possible_pos;
        S* dst_score = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
        D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
        occupy_result = OccupyResult::DUPLICATE;
        dst_digest[0] = get_digest<K>(key);
        dst_score[0] = scores == nullptr ? device_nano<S>() : score;
        break;
      } else if (bucket_size == BUCKET_SIZE) {
        continue;
      }
      uint32_t empty_digests = 
                __byte_perm(empty_digest<K>(), empty_digest<K>(), 0x0000);
      find_result = __vcmpeq4(probing_digests, empty_digests);
      find_result &= 0x01010101;
      do {
        if (find_result == 0) break;
        // Little endian.
        uint32_t index = (__ffs(find_result) - 1) >> 3;
        find_result &= (find_result - 1);
        possible_pos = pos_cur + i * 4 + index;
        if (offset == 0 && possible_pos < key_pos) continue;
        auto current_key = BKT::keys(bucket_keys_ptr, possible_pos);
        K expected_key = static_cast<K>(EMPTY_KEY);
        // __threadfence();
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
        // __threadfence();
      } while (!result);
      if (result) {
        key_pos = possible_pos;
        S* dst_score = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
        D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
        occupy_result = OccupyResult::OCCUPIED_EMPTY;
        dst_digest[0] = get_digest<K>(key);
        dst_score[0] = scores == nullptr ? device_nano<S>() : score;
        atomicAdd(bucket_size_address, 1);
        break;
      }
    }
  }
  if (occupy_result == OccupyResult::INITIAL) {
    evict_idx = atomicAdd(evict_number, 1);
  }
  while (occupy_result == OccupyResult::INITIAL) {
    int min_pos = -1;
    S* src = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, 0);
    S min_score = MAX_SCORE;
    for (int j = 0; j < BUCKET_SIZE; j += 1) {
      S temp_score = src[j];
      if (temp_score < min_score) {
        min_score = temp_score;
        min_pos = j;
      }
    }
    if (score < min_score) {
      occupy_result = OccupyResult::REFUSED;
      evicted_keys[evict_idx] = key;
      if (evicted_scores != nullptr) {
        evicted_scores[evict_idx] = score;
      }
      break;
    }
    auto min_score_key = BKT::keys(bucket_keys_ptr, min_pos);
    // __threadfence();
    auto expected_key = min_score_key->load(cuda::std::memory_order_acquire);
    // __threadfence();
    if (expected_key != static_cast<K>(LOCKED_KEY) &&
      expected_key != static_cast<K>(EMPTY_KEY)) {
      S* min_score_ptr = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos);
      // __threadfence();
      bool result = min_score_key->compare_exchange_strong(
        expected_key, static_cast<K>(LOCKED_KEY),
        cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
      // __threadfence();
      if (result) {
        auto verify_score_ptr = reinterpret_cast<AtomicScore<S>*>(min_score_ptr);
        auto verify_score = verify_score_ptr->load(cuda::std::memory_order_acquire);
        if (verify_score > min_score) {
          // __threadfence();
          min_score_key->store(expected_key,
                      cuda::std::memory_order_release);
          // __threadfence();
        } else {
          key_pos = min_pos;
          *min_score_ptr = scores == nullptr ? device_nano<S>() : score;
          D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
          *dst_digest = get_digest<K>(key);
          if (expected_key == static_cast<K>(RECLAIM_KEY)) {
            atomicAdd(bucket_size_address, 1);
            occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
          } else {
            occupy_result = OccupyResult::EVICT;
            evicted_keys[evict_idx] = expected_key;
            if (evicted_scores != nullptr) {
              evicted_scores[evict_idx] = min_score;
            }
          }
        }
      }
    }
  }
  V* bucket_values_ptr = bucket->values() + key_pos * dim;
  const V* param_values_ptr = values + kv_idx * dim;
  V* evicted_values_ptr = evicted_values + evict_idx * dim;
  for (int i = 0; i < dim; i += 4) {
    if (occupy_result == OccupyResult::REFUSED) {
      ((float4*)(evicted_values_ptr + i))[0] = ((float4*)(param_values_ptr + i))[0];
    } else {
      if (occupy_result == OccupyResult::EVICT) {
        ((float4*)(evicted_values_ptr + i))[0] = ((float4*)(bucket_values_ptr + i))[0];
      }
      ((float4*)(bucket_values_ptr + i))[0] = ((float4*)(param_values_ptr + i))[0];
    }
  }
  // __threadfence();
  auto key_address = BKT::keys(bucket_keys_ptr, key_pos);
  key_address->store(key, cuda::std::memory_order_release);
}

template <
typename K = uint64_t,
typename V = float,
typename S = uint64_t,
typename VecV = float4,
typename D = uint8_t,
typename VecD = uint4,
uint32_t BUCKET_SIZE = 128,
uint32_t BLOCK_SIZE = 64,
uint32_t GROUP_SIZE = 8,
uint32_t BUF_V = 64>
__global__ void upsert_and_evict_kernel_unique_v1(
  Bucket<K, V, S>* buckets, int32_t* buckets_size, const uint64_t buckets_num, 
  const uint32_t dim, const K* __restrict keys, const V* __restrict values, 
  const S* __restrict scores, K* __restrict evicted_keys, V* __restrict evicted_values,
  S* __restrict evicted_scores, uint32_t n, size_t* evict_number) {
  using BKT = Bucket<K, V, S>;
  constexpr uint32_t VecD_SIZE = sizeof(VecD) / sizeof(D);
  constexpr uint32_t GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;

  __shared__ S sm_param_scores[BLOCK_SIZE];
  __shared__ V* sm_values_address[BLOCK_SIZE];
  __shared__ int* sm_buckets_size_address[BLOCK_SIZE];

  __shared__ D sm_bucket_digests[GROUP_NUM][2][BUCKET_SIZE];
  __shared__ S sm_bucket_scores[GROUP_NUM][2][BUCKET_SIZE];
  __shared__ V sm_values_buffer[GROUP_NUM][2][BUF_V * 2];

  // Initialization.
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  uint32_t tx = threadIdx.x;
  uint32_t kv_idx = blockIdx.x * blockDim.x + tx;
  K key;
  uint32_t target_digests;
  K* bucket_keys_ptr {nullptr};
  OccupyResult occupy_result{OccupyResult::INITIAL};
  uint32_t key_pos = 0;
  uint32_t evict_idx = 0;
  if (kv_idx < n) {
    key = keys[kv_idx];
    if (scores != nullptr) {
      __pipeline_memcpy_async(sm_param_scores + tx, scores + kv_idx, sizeof(S));
    } else {
      sm_param_scores[tx] = static_cast<S>(MAX_SCORE);
    }
    if (!IS_RESERVED_KEY(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      D digest = digest_from_hashed<K>(hashed_key);
      target_digests = __byte_perm(digest, digest, 0x0000);
      uint64_t global_idx = static_cast<uint64_t>(
          hashed_key % (buckets_num * BUCKET_SIZE));
      uint64_t bkt_idx = global_idx / BUCKET_SIZE;
      sm_buckets_size_address[tx] = buckets_size + bkt_idx;
      key_pos = global_idx & (BUCKET_SIZE - 1);
      Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys();
      __pipeline_memcpy_async(sm_values_address + tx, &(bucket->values()),
                               sizeof(V*));
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }
  } else {
    occupy_result = OccupyResult::ILLEGAL;
  }

  uint32_t rank = g.thread_rank();
  uint32_t groupID = threadIdx.x / GROUP_SIZE;
  // Pipeline loading.
  auto occupy_result_next = g.shfl(occupy_result, 0);
  if (occupy_result_next == OccupyResult::INITIAL) {
    D* dst = sm_bucket_digests[groupID][0] + rank * VecD_SIZE;
    auto keys_ptr_next = g.shfl(bucket_keys_ptr, 0);
    D* src = BKT::digests(keys_ptr_next, BUCKET_SIZE, rank * VecD_SIZE);
    if (rank * VecD_SIZE < BUCKET_SIZE) {
      __pipeline_memcpy_async(dst, src, sizeof(VecD));
    }
  }
  __pipeline_commit();
  __pipeline_commit();
  __pipeline_commit();
  for (int32_t i = 0; i < GROUP_SIZE; i++) {
    if (i + 1 < GROUP_SIZE) {
      auto occupy_result_next = g.shfl(occupy_result, i + 1);
      if (occupy_result_next == OccupyResult::INITIAL) {
        D* dst = sm_bucket_digests[groupID][diff_buf(i)] + rank * VecD_SIZE;
        auto keys_ptr_next = g.shfl(bucket_keys_ptr, i + 1);
        D* src = BKT::digests(keys_ptr_next, BUCKET_SIZE, rank * VecD_SIZE);
        if (rank * VecD_SIZE < BUCKET_SIZE) {
          __pipeline_memcpy_async(dst, src, sizeof(VecD));
        }
      }
    }
    __pipeline_commit();
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur == OccupyResult::INITIAL) {
      D* src = sm_bucket_digests[groupID][same_buf(i)];
      auto target_digests_cur = g.shfl(target_digests, i);
      uint32_t tx_cur = groupID * GROUP_SIZE + i;
      auto bucket_size_address = sm_buckets_size_address[tx_cur];
      auto bucket_size_cur = bucket_size_address[0];
      K key_cur = g.shfl(key, i);
      auto start_pos_cur = g.shfl(key_pos, i);
      auto keys_ptr_cur = g.shfl(bucket_keys_ptr, i);
      __pipeline_wait_prior(3);
      uint32_t start_offset = start_pos_cur / 4;
      uint32_t probing_digests;
      constexpr uint32_t LEN = BUCKET_SIZE / 4;
      uint32_t found_vote = 0;
      for (int k = rank; k < LEN + GROUP_SIZE; k += GROUP_SIZE) {
        uint32_t probing_offset = 4 * ((start_offset + k) & (LEN - 1));
        probing_digests = reinterpret_cast<uint32_t*>(src + probing_offset)[0];
        uint32_t find_result = __vcmpeq4(probing_digests, target_digests_cur);
        find_result &= 0x01010101;
        uint32_t possible_pos = 0;
        bool result = false;
        do {
          if (find_result == 0) break;
          // Little endian.
          int32_t index = (__ffs(find_result) - 1) >> 3;
          find_result &= (find_result - 1);
          possible_pos = probing_offset + index;
          auto current_key = BKT::keys(keys_ptr_cur, possible_pos);
          K expected_key = key_cur;
          result = current_key->compare_exchange_strong(
              expected_key, static_cast<K>(LOCKED_KEY),
              cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
        } while (!result);
        found_vote = g.ballot(result);
        if (found_vote) {
          int32_t src_lane = __ffs(found_vote) - 1;
          possible_pos = g.shfl(possible_pos, src_lane);
          if (rank == i) {
            S score = scores == nullptr ? device_nano<S>() : sm_param_scores[tx];
            key_pos = possible_pos;
            S* dst_score = (S*)BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
            D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
            occupy_result = OccupyResult::DUPLICATE;
            dst_digest[0] = get_digest<K>(key);
            dst_score[0] = score;
          }
          break;
        } else if (bucket_size_cur < BUCKET_SIZE) {
          uint32_t empty_digests = 
                    __byte_perm(empty_digest<K>(), empty_digest<K>(), 0x0000);
          uint32_t find_result = __vcmpeq4(probing_digests, empty_digests);
          find_result &= 0x01010101;
          uint32_t possible_pos = 0;
          bool result = false;
          uint32_t found_vote = 0;
          for (int32_t offset = 0; offset < GROUP_SIZE; offset += 1) {
            if (rank == offset) {
              do {
                if (find_result == 0) break;
                // Little endian.
                int32_t index = (__ffs(find_result) - 1) >> 3;
                find_result &= (find_result - 1);
                possible_pos = probing_offset + index;
                if (k == 0 && offset == 0 && possible_pos < start_pos_cur) continue;
                auto current_key = BKT::keys(keys_ptr_cur, possible_pos);
                K expected_key = static_cast<K>(EMPTY_KEY);
                result = current_key->compare_exchange_strong(
                    expected_key, static_cast<K>(LOCKED_KEY),
                    cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
              } while (!result);
            }
            found_vote = g.ballot(result);
            if (found_vote) {
              int32_t src_lane = __ffs(found_vote) - 1;
              possible_pos = g.shfl(possible_pos, src_lane);
              if (rank == i) {
                S score = scores == nullptr ? device_nano<S>() : sm_param_scores[tx];
                int* bucket_size_address = sm_buckets_size_address[tx];
                key_pos = possible_pos;
                S* dst_score = (S*)BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
                D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
                occupy_result = OccupyResult::OCCUPIED_EMPTY;
                dst_digest[0] = get_digest<K>(key);
                dst_score[0] = score;
                atomicAdd(bucket_size_address, 1);
              }
              break;
            }
          }
          if (found_vote) {
            break;
          }
        }
      }
      occupy_result_cur = g.shfl(occupy_result, i);
      if (occupy_result_cur == OccupyResult::INITIAL) {
        if (rank == i) {
          evict_idx = atomicAdd(evict_number, 1);
        }
        S* dst = sm_bucket_scores[groupID][same_buf(i)] + rank * 2;
        S* src = (S*)BKT::scores(keys_ptr_cur, BUCKET_SIZE, rank * 2);
        #pragma unroll
        for (int32_t k = 0; k < BUCKET_SIZE; k += GROUP_SIZE * 2) {
          __pipeline_memcpy_async(dst + k, src + k, sizeof(S) * 2);
        }
      }
    }
    __pipeline_commit();
    if (i > 0) {
      occupy_result_cur = g.shfl(occupy_result, i - 1);
      uint32_t tx_cur = groupID * GROUP_SIZE + i - 1;
      S score_cur = scores == nullptr ? device_nano<S>() : sm_param_scores[tx_cur];
      auto bucket_size_address = sm_buckets_size_address[tx_cur];
      __pipeline_wait_prior(3);
      S* src = sm_bucket_scores[groupID][diff_buf(i)];
      while (occupy_result_cur == OccupyResult::INITIAL) {
        int min_pos_local = -1;
        S min_score_local = MAX_SCORE;
        for (int j = rank; j < BUCKET_SIZE; j += GROUP_SIZE) {
          S temp_score = src[j];
          if (temp_score < min_score_local) {
            min_score_local = temp_score;
            min_pos_local = j;
          }
        }
        const S min_score_global =
            cg::reduce(g, min_score_local, cg::less<S>());
        if (score_cur < min_score_global) {
          if (rank == i - 1) {
            occupy_result = OccupyResult::REFUSED;
            if (evicted_scores != nullptr && scores != nullptr) {
              evicted_scores[evict_idx] = score_cur;
            }
            evicted_keys[evict_idx] = key;
          }
          occupy_result_cur = g.shfl(occupy_result, i - 1);
          break;
        }
        uint32_t vote = g.ballot(min_score_local <= min_score_global);
        if (vote) {
          int src_lane = __ffs(vote) - 1;
          int min_pos_global = g.shfl(min_pos_local, src_lane);
          if (rank == i - 1) {
            src[min_pos_global] = static_cast<S>(MAX_SCORE); // Mark visited.
            auto min_score_key = BKT::keys(bucket_keys_ptr, min_pos_global);
            auto expected_key = min_score_key->load(cuda::std::memory_order_acquire);
            if (expected_key != static_cast<K>(LOCKED_KEY) &&
              expected_key != static_cast<K>(EMPTY_KEY)) {
              S* min_score_ptr = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos_global);
              bool result = min_score_key->compare_exchange_strong(
                expected_key, static_cast<K>(LOCKED_KEY),
                cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
              if (result) {
                if (min_score_ptr[0] > min_score_global) {
                  min_score_key->store(expected_key,
                              cuda::std::memory_order_release);
                } else {
                  if (expected_key == static_cast<K>(RECLAIM_KEY)) {
                    atomicAdd(bucket_size_address, 1);
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                  } else {
                    occupy_result = OccupyResult::EVICT;
                    evicted_keys[evict_idx] = expected_key;
                    if (evicted_scores != nullptr) {
                      evicted_scores[evict_idx] = min_score_global;
                    }
                  }
                  key_pos = min_pos_global;
                  S* dst_score = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
                  dst_score[0] = score_cur;
                  D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
                  dst_digest[0] = get_digest<K>(key);
                }
              }
            }
          }
          occupy_result_cur = g.shfl(occupy_result, i - 1);
        }
      }
      if (occupy_result_cur != OccupyResult::ILLEGAL) {
        auto values_address_cur = sm_values_address[groupID * GROUP_SIZE + i - 1];
        auto key_pos_cur = g.shfl(key_pos, i - 1);
        auto kv_idx_cur = g.shfl(kv_idx, i - 1);

        const V* src = values + kv_idx_cur * dim;
        V* dst = sm_values_buffer[groupID][diff_buf(i)];
        int rank_offset = rank * 4;
        if (rank_offset < dim) {
          __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
        }
        if (occupy_result_cur == OccupyResult::EVICT) {
          const V* src = values_address_cur + key_pos_cur * dim;
          V* dst = sm_values_buffer[groupID][diff_buf(i)] + BUF_V;
          if (rank_offset < dim) {
            __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
          }
        }
      }
    }
    __pipeline_commit();
  
    if (i > 1) {
      occupy_result_cur = g.shfl(occupy_result, i - 2);
      if (occupy_result_cur != OccupyResult::ILLEGAL) {
        auto values_address_cur = sm_values_address[groupID * GROUP_SIZE + i - 2];
        auto key_pos_cur = g.shfl(key_pos, i - 2);
        auto evict_idx_cur = g.shfl(evict_idx, i - 2);
        int rank_offset = rank * 4;

        if (occupy_result_cur == OccupyResult::REFUSED) {
          V* src = sm_values_buffer[groupID][same_buf(i)];
          V* dst = evicted_values + evict_idx_cur * dim;
          __pipeline_wait_prior(3);
          if (rank_offset < dim) {
            float4 vecV = ((float4*)(src + rank_offset))[0];
            ((float4*)(dst + rank_offset))[0] = vecV;
          }
          continue;
        } else {
          V* src = sm_values_buffer[groupID][same_buf(i)];
          V* dst = values_address_cur + key_pos_cur * dim;
          __pipeline_wait_prior(3);
          if (rank_offset < dim) {
            float4 vecV = ((float4*)(src + rank_offset))[0];
            ((float4*)(dst + rank_offset))[0] = vecV;
          }
          if (rank == i - 2) {
            auto key_address = BKT::keys(bucket_keys_ptr, key_pos);
            key_address->store(key, cuda::std::memory_order_release);
          }
        }
        if (occupy_result_cur == OccupyResult::EVICT) {
          V* src = sm_values_buffer[groupID][same_buf(i)] + BUF_V;
          V* dst = evicted_values + evict_idx_cur * dim;
          __pipeline_wait_prior(3);
          if (rank_offset < dim) {
            float4 vecV = ((float4*)(src + rank_offset))[0];
            ((float4*)(dst + rank_offset))[0] = vecV;
          }
        }
      }
    }
  }
  auto occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
  uint32_t tx_cur = groupID * GROUP_SIZE + GROUP_SIZE - 1;
  S score_cur = sm_param_scores[tx_cur];
  auto bucket_size_address = sm_buckets_size_address[tx_cur];
  __pipeline_wait_prior(1);
  S* src = sm_bucket_scores[groupID][diff_buf(GROUP_SIZE)];
  while (occupy_result_cur == OccupyResult::INITIAL) {
    int min_pos_local = -1;
    S min_score_local = MAX_SCORE;
    for (int j = rank; j < BUCKET_SIZE; j += GROUP_SIZE) {
      S temp_score = src[j];
      if (temp_score < min_score_local) {
        min_score_local = temp_score;
        min_pos_local = j;
      }
    }
    const S min_score_global =
        cg::reduce(g, min_score_local, cg::less<S>());
    if (score_cur < min_score_global) {
      if (rank == GROUP_SIZE - 1) {
        occupy_result = OccupyResult::REFUSED;
        if (evicted_scores != nullptr && scores != nullptr) {
          evicted_scores[evict_idx] = score_cur;
        }
        evicted_keys[evict_idx] = key;
      }
      occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
      break;
    }
    uint32_t vote = g.ballot(min_score_local <= min_score_global);
    if (vote) {
      int src_lane = __ffs(vote) - 1;
      int min_pos_global = g.shfl(min_pos_local, src_lane);
      if (rank == GROUP_SIZE - 1) {
        src[min_pos_global] = MAX_SCORE; // Mark visited.
        auto min_score_key = BKT::keys(bucket_keys_ptr, min_pos_global);
        auto expected_key = min_score_key->load(cuda::std::memory_order_acquire);
        if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
          auto min_score_ptr = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, min_pos_global);
          bool result = min_score_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_acquire);
          if (result) {
            if (min_score_ptr[0] > min_score_global) {
              min_score_key->store(expected_key,
                          cuda::std::memory_order_release);
            } else {
              if (expected_key == static_cast<K>(RECLAIM_KEY)) {
                atomicAdd(bucket_size_address, 1);
                occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
              } else {
                occupy_result = OccupyResult::EVICT;
                evicted_keys[evict_idx] = expected_key;
                if (evicted_scores != nullptr) {
                  evicted_scores[evict_idx] = min_score_global;
                }
              }
              key_pos = min_pos_global;
              S* dst_score = BKT::scores(bucket_keys_ptr, BUCKET_SIZE, key_pos);
              dst_score[0] = score_cur;
              D* dst_digest = BKT::digests(bucket_keys_ptr, BUCKET_SIZE, key_pos);
              dst_digest[0] = get_digest<K>(key);
            }
          }
        }
      }
      occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
    }
  }
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    auto values_address_cur = sm_values_address[groupID * GROUP_SIZE + GROUP_SIZE - 1];
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);
    auto kv_idx_cur = g.shfl(kv_idx, GROUP_SIZE - 1);

    const V* src = values + kv_idx_cur * dim;
    V* dst = sm_values_buffer[groupID][diff_buf(GROUP_SIZE)];
    int rank_offset = rank * 4;
    if (rank_offset < dim) {
      __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
    }
    if (occupy_result_cur == OccupyResult::EVICT) {
      const V* src = values_address_cur + key_pos_cur * dim;
      V* dst = sm_values_buffer[groupID][diff_buf(GROUP_SIZE)] + BUF_V;
      if (rank_offset < dim) {
        __pipeline_memcpy_async(dst + rank_offset, src + rank_offset, sizeof(float4));
      }
    }
  }
  __pipeline_commit();

  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 2);
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    auto values_address_cur = sm_values_address[groupID * GROUP_SIZE + GROUP_SIZE - 2];
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 2);
    auto evict_idx_cur = g.shfl(evict_idx, GROUP_SIZE - 2);
    int rank_offset = rank * 4;

    if (occupy_result_cur == OccupyResult::REFUSED) {
      V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE)];
      V* dst = evicted_values + evict_idx_cur * dim;
      __pipeline_wait_prior(1);
      if (rank_offset < dim) {
        float4 vecV = ((float4*)(src + rank_offset))[0];
        ((float4*)(dst + rank_offset))[0] = vecV;
      }
    } else {
      V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE)];
      V* dst = values_address_cur + key_pos_cur * dim;
      __pipeline_wait_prior(1);
      if (rank_offset < dim) {
        float4 vecV = ((float4*)(src + rank_offset))[0];
        ((float4*)(dst + rank_offset))[0] = vecV;
      }
      if (rank == GROUP_SIZE - 2) {
        auto key_address = BKT::keys(bucket_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
      if (occupy_result_cur == OccupyResult::EVICT) {
        V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE)] + BUF_V;
        V* dst = evicted_values + evict_idx_cur * dim;
        __pipeline_wait_prior(1);
        if (rank_offset < dim) {
          float4 vecV = ((float4*)(src + rank_offset))[0];
          ((float4*)(dst + rank_offset))[0] = vecV;
        }
      }
    }
  }

  occupy_result_cur = g.shfl(occupy_result, GROUP_SIZE - 1);
  if (occupy_result_cur != OccupyResult::ILLEGAL) {
    auto values_address_cur = sm_values_address[groupID * GROUP_SIZE + GROUP_SIZE - 1];
    auto key_pos_cur = g.shfl(key_pos, GROUP_SIZE - 1);
    auto evict_idx_cur = g.shfl(evict_idx, GROUP_SIZE - 1);
    int rank_offset = rank * 4;

    if (occupy_result_cur == OccupyResult::REFUSED) {
      V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE + 1)];
      V* dst = evicted_values + evict_idx_cur * dim;
      __pipeline_wait_prior(0);
      if (rank_offset < dim) {
        float4 vecV = ((float4*)(src + rank_offset))[0];
        ((float4*)(dst + rank_offset))[0] = vecV;
      }
    } else {
      V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE + 1)];
      V* dst = values_address_cur + key_pos_cur * dim;
      __pipeline_wait_prior(0);
      if (rank_offset < dim) {
        float4 vecV = ((float4*)(src + rank_offset))[0];
        ((float4*)(dst + rank_offset))[0] = vecV;
      }
      if (rank == GROUP_SIZE - 1) {
        auto key_address = BKT::keys(bucket_keys_ptr, key_pos);
        key_address->store(key, cuda::std::memory_order_release);
      }
      if (occupy_result_cur == OccupyResult::EVICT) {
        V* src = sm_values_buffer[groupID][same_buf(GROUP_SIZE + 1)] + BUF_V;
        V* dst = evicted_values + evict_idx_cur * dim;
        __pipeline_wait_prior(0);
        if (rank_offset < dim) {
          float4 vecV = ((float4*)(src + rank_offset))[0];
          ((float4*)(dst + rank_offset))[0] = vecV;
        }
      }
    }
  }
}

template <typename K, typename V, typename S = uint64_t,
          typename ArchTag = Sm80>
struct UpsertAndEvictKernelSelector {
  using ValueDimConfig = LookupValueDimConfig<ArchTag>;

  static inline bool callable(uint32_t& bucket_size, uint32_t value_size) {
    
  }

  static inline uint32_t max_value_size() {
    return ValueDimConfig::pipeline_v1_dim * sizeof(float);
  }

  static void select_kernel(LookupKernelParams<K, V, S>& params,
                            cudaStream_t& stream) {
    constexpr int BUCKET_SIZE = 128;
    constexpr size_t ValueGranularity = sizeof(float);
    constexpr uint32_t v1_value_dim = ValueDimConfig::pipeline_v1_dim;
    constexpr uint32_t v2_value_dim = ValueDimConfig::pipeline_v2_dim;

    params.dim =
        static_cast<uint32_t>(params.dim * sizeof(V) / ValueGranularity);

    if (params.scores == nullptr) {
      using CopyScore = CopyScoreEmpty<S, K, BUCKET_SIZE>;
      if (params.dim > v2_value_dim) {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float4,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float2,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV1<K, V, S, CopyScore, float,
                                  v1_value_dim>::launch_kernel(params, stream);
        }
      } else {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float4,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float2,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV2<K, V, S, CopyScore, float,
                                  v2_value_dim>::launch_kernel(params, stream);
        }
      }
    } else {
      using CopyScore = CopyScoreByPassCache<S, K, BUCKET_SIZE>;
      if (params.dim > v2_value_dim) {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float4,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV1<K, V, S, CopyScore, float2,
                                  v1_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV1<K, V, S, CopyScore, float,
                                  v1_value_dim>::launch_kernel(params, stream);
        }
      } else {
        if (params.dim % 4 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float4,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else if (params.dim % 2 == 0) {
          SelectLookupCopyValueV2<K, V, S, CopyScore, float2,
                                  v2_value_dim>::launch_kernel(params, stream);
        } else {
          static_assert(sizeof(V) == 4);
          SelectLookupCopyValueV2<K, V, S, CopyScore, float,
                                  v2_value_dim>::launch_kernel(params, stream);
        }
      }
    }
  }  // End function
};

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
      evicted_keys[key_idx] = insert_key;
      evicted_scores[key_idx] = insert_score;
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