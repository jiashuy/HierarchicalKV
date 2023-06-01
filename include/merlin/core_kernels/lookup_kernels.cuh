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

// Using 32 threads to deal with one key
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename CopyScore = CopyScoreEmpty<S, K, 128>,
          typename CopyValue = CopyValueTwoGroup<float, float4, 32>>
__global__ void lookup_kernel_with_io_pipeline_v1(
    const Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
    bool* __restrict founds, size_t n) {
  constexpr int GROUP_SIZE = 32;
  constexpr int RESERVE = 16;
  constexpr int DIM_BUF = 224;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr int DIGEST_SPAN = BUCKET_SIZE / 4;

  __shared__ int sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr[BLOCK_SIZE];
  __shared__ V* sm_values_ptr[BLOCK_SIZE];
  // Reuse
  S* sm_target_scores = reinterpret_cast<S*>(sm_target_keys);
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];
  __shared__ V sm_vector[2][GROUP_NUM][DIM_BUF];

  // Initialization
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int groupID = threadIdx.x / GROUP_SIZE;
  int rank = g.thread_rank();
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;
  if (key_idx_base >= n) return;
  int loop_num =
      (n - key_idx_base) < GROUP_SIZE ? (n - key_idx_base) : GROUP_SIZE;
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    K target_key = keys[key_idx_base + rank];
    sm_target_keys[idx_block] = target_key;
    K hashed_key = Murmur3HashDevice(target_key);
    uint8_t target_digest = static_cast<uint8_t>(hashed_key >> 32);
    sm_target_digests[idx_block] = static_cast<uint32_t>(target_digest);
    int global_idx = hashed_key % (buckets_num * BUCKET_SIZE);
    int bkt_idx = global_idx / BUCKET_SIZE;
    const Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __pipeline_memcpy_async(sm_keys_ptr + idx_block, &(bucket->keys_),
                            sizeof(K*));
    __pipeline_commit();
    __pipeline_memcpy_async(sm_values_ptr + idx_block, &(bucket->vectors),
                            sizeof(V*));
  }
  __pipeline_wait_prior(0);

  // Pipeline loading
  S* scores_ptr =
      reinterpret_cast<S*>(sm_keys_ptr[groupID * GROUP_SIZE] + BUCKET_SIZE);
  uint8_t* digests_ptr = reinterpret_cast<uint8_t*>(scores_ptr + BUCKET_SIZE);
  __pipeline_memcpy_async(sm_probing_digests[0] + groupID * DIGEST_SPAN + rank,
                          digests_ptr + rank * 4, sizeof(uint32_t));
  __pipeline_commit();
  __pipeline_commit();  // padding
  __pipeline_commit();  // padding

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    /* Step1: prefetch all digests in one bucket */
    if ((i + 1) < loop_num) {
      S* scores_ptr =
          reinterpret_cast<S*>(sm_keys_ptr[key_idx_block + 1] + BUCKET_SIZE);
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(scores_ptr + BUCKET_SIZE);
      __pipeline_memcpy_async(
          sm_probing_digests[DIFF_BUF(i)] + groupID * DIGEST_SPAN + rank,
          digests_ptr + rank * 4, sizeof(uint32_t));
    }
    __pipeline_commit();

    /* Step2: check digests and load possible keys */
    uint32_t target_digest = sm_target_digests[key_idx_block];
    uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(3);
    uint32_t probing_digests =
        sm_probing_digests[SAME_BUF(i)][groupID * DIGEST_SPAN + rank];
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests);
    uint32_t find_result = 0;
    if ((find_result_ & 0x01) != 0) find_result |= 0x01;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x02;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x04;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x08;
    int find_number = __popc(find_result);
    int group_base = 0;
    if (find_number > 0) {
      group_base = atomicAdd(sm_counts + key_idx_block, find_number);
    }
    bool gt_reserve = (group_base + find_number) > RESERVE;
    int gt_vote = g.ballot(gt_reserve);
    K* key_ptr = sm_keys_ptr[key_idx_block];
    if (gt_vote == 0) {
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = rank * 4 + digest_idx;
          sm_possible_pos[SAME_BUF(i)][groupID * RESERVE + group_base] =
              key_pos;
          __pipeline_memcpy_async(
              sm_possible_keys[SAME_BUF(i)] + (groupID * RESERVE + group_base),
              key_ptr + key_pos, sizeof(K));
          group_base += 1;
        } else {
          break;
        }
      } while (true);
    } else {
      K target_key = sm_target_keys[key_idx_block];
      sm_counts[key_idx_block] = 0;
      int found_vote = 0;
      bool found = false;
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = rank * 4 + digest_idx;
          K possible_key = key_ptr[key_pos];
          if (possible_key == target_key) {
            found = true;
            sm_counts[key_idx_block] = 1;
            sm_possible_pos[SAME_BUF(i)][groupID * RESERVE] = key_pos;
            sm_possible_keys[SAME_BUF(i)][groupID * RESERVE] = possible_key;
          }
        }
        found_vote = g.ballot(found);
        if (found_vote) {
          break;
        }
        found_vote = digest_idx >= 0;
      } while (g.any(found_vote));
    }
    __pipeline_commit();

    /* Step3: check possible keys, and prefecth the value and score */
    if (i > 0) {
      key_idx_block -= 1;
      K target_key = sm_target_keys[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      sm_founds[key_idx_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
      V* value_ptr = sm_values_ptr[key_idx_block];
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[DIFF_BUF(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[DIFF_BUF(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
          CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                             score_ptr + key_pos);
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        V* v_dst = sm_vector[DIFF_BUF(i)][groupID];
        sm_founds[key_idx_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        V* v_src = value_ptr + target_pos * dim;
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    /* Step4: write back value and score */
    if (i > 1) {
      key_idx_block -= 1;
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      V* v_src = sm_vector[SAME_BUF(i)][groupID];
      V* v_dst = values + key_idx_grid * dim;
      int found_flag = sm_founds[key_idx_block];
      __pipeline_wait_prior(3);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        founds[key_idx_grid] = true;
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
    V* value_ptr = sm_values_ptr[key_idx_block];
    __pipeline_wait_prior(1);
    int key_pos;
    bool found_flag = false;
    if (rank < possible_num) {
      key_pos = sm_possible_pos[DIFF_BUF(loop_num)][groupID * RESERVE + rank];
      K possible_key =
          sm_possible_keys[DIFF_BUF(loop_num)][groupID * RESERVE + rank];
      if (target_key == possible_key) {
        found_flag = true;
        CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                           score_ptr + key_pos);
      }
    }
    int found_vote = g.ballot(found_flag);
    if (found_vote) {
      sm_founds[key_idx_block] = 1;
      int src_lane = __ffs(found_vote) - 1;
      int target_pos = g.shfl(key_pos, src_lane);
      V* v_src = value_ptr + target_pos * dim;
      V* v_dst = sm_vector[DIFF_BUF(loop_num)][groupID];
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  /* Pipeline emptying: step4, i = loop_num */
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_src = sm_vector[SAME_BUF(loop_num)][groupID];
    V* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(1);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  /* Pipeline emptying: step4, i = loop_num + 1 */
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_src = sm_vector[SAME_BUF(loop_num + 1)][groupID];
    V* v_dst = values + key_idx_grid * dim;
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(0);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }
}  // End function

// Using 16 threads to deal with one key
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename CopyScore = CopyScoreEmpty<S, K, 128>,
          typename CopyValue = CopyValueTwoGroup<float, float4, 16>>
__global__ void lookup_kernel_with_io_pipeline_v2(
    const Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V* __restrict values, S* __restrict scores,
    bool* __restrict founds, size_t n) {
  constexpr int GROUP_SIZE = 16;
  constexpr int RESERVE = 8;
  constexpr int DIM_BUF = 128;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BUCKET_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  constexpr int DIGEST_SPAN = BUCKET_SIZE / 4;

  __shared__ int sm_target_digests[BLOCK_SIZE];
  __shared__ K sm_target_keys[BLOCK_SIZE];
  __shared__ K* sm_keys_ptr[BLOCK_SIZE];
  __shared__ V* sm_values_ptr[BLOCK_SIZE];
  // Reuse
  S* sm_target_scores = reinterpret_cast<S*>(sm_target_keys);
  int* sm_counts = sm_target_digests;
  int* sm_founds = sm_counts;
  // Double buffer
  __shared__ uint32_t sm_probing_digests[2][GROUP_NUM * DIGEST_SPAN];
  __shared__ K sm_possible_keys[2][GROUP_NUM * RESERVE];
  __shared__ int sm_possible_pos[2][GROUP_NUM * RESERVE];
  __shared__ V sm_vector[2][GROUP_NUM][DIM_BUF];

  // Initialization
  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int groupID = threadIdx.x / GROUP_SIZE;
  int rank = g.thread_rank();
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;
  if (key_idx_base >= n) return;
  int loop_num =
      (n - key_idx_base) < GROUP_SIZE ? (n - key_idx_base) : GROUP_SIZE;
  if (rank < loop_num) {
    int idx_block = groupID * GROUP_SIZE + rank;
    K target_key = keys[key_idx_base + rank];
    sm_target_keys[idx_block] = target_key;
    K hashed_key = Murmur3HashDevice(target_key);
    uint8_t target_digest = static_cast<uint8_t>(hashed_key >> 32);
    sm_target_digests[idx_block] = static_cast<uint32_t>(target_digest);
    int global_idx = hashed_key % (buckets_num * BUCKET_SIZE);
    int bkt_idx = global_idx / BUCKET_SIZE;
    const Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __pipeline_memcpy_async(sm_keys_ptr + idx_block, &(bucket->keys_),
                            sizeof(K*));
    __pipeline_commit();
    __pipeline_memcpy_async(sm_values_ptr + idx_block, &(bucket->vectors),
                            sizeof(V*));
  }
  __pipeline_wait_prior(0);

  // Pipeline loading
  S* scores_ptr =
      reinterpret_cast<S*>(sm_keys_ptr[groupID * GROUP_SIZE] + BUCKET_SIZE);
  uint8_t* digests_ptr = reinterpret_cast<uint8_t*>(scores_ptr + BUCKET_SIZE);
  __pipeline_memcpy_async(
      sm_probing_digests[0] + groupID * DIGEST_SPAN + rank * 2,
      digests_ptr + rank * 8, sizeof(uint2));
  __pipeline_commit();
  __pipeline_commit();  // padding
  __pipeline_commit();  // padding

  for (int i = 0; i < loop_num; i++) {
    int key_idx_block = groupID * GROUP_SIZE + i;

    /* Step1: prefetch all digests in one bucket */
    if ((i + 1) < loop_num) {
      S* scores_ptr =
          reinterpret_cast<S*>(sm_keys_ptr[key_idx_block + 1] + BUCKET_SIZE);
      uint8_t* digests_ptr =
          reinterpret_cast<uint8_t*>(scores_ptr + BUCKET_SIZE);
      __pipeline_memcpy_async(
          sm_probing_digests[DIFF_BUF(i)] + groupID * DIGEST_SPAN + rank * 2,
          digests_ptr + rank * 8, sizeof(uint2));
    }
    __pipeline_commit();

    /* Step2: check digests and load possible keys */
    uint32_t target_digest = sm_target_digests[key_idx_block];
    uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);
    sm_counts[key_idx_block] = 0;
    __pipeline_wait_prior(3);
    uint32_t probing_digests =
        sm_probing_digests[SAME_BUF(i)][groupID * DIGEST_SPAN + rank];
    uint32_t find_result_ = __vcmpeq4(probing_digests, target_digests);
    uint32_t find_result = 0;
    if ((find_result_ & 0x01) != 0) find_result |= 0x01;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x02;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x04;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x08;
    probing_digests = sm_probing_digests[SAME_BUF(i)][groupID * DIGEST_SPAN +
                                                      rank + GROUP_SIZE];
    find_result_ = __vcmpeq4(probing_digests, target_digests);
    if ((find_result_ & 0x01) != 0) find_result |= 0x10;
    if ((find_result_ & 0x0100) != 0) find_result |= 0x20;
    if ((find_result_ & 0x010000) != 0) find_result |= 0x40;
    if ((find_result_ & 0x01000000) != 0) find_result |= 0x80;
    int find_number = __popc(find_result);
    int group_base = 0;
    if (find_number > 0) {
      group_base = atomicAdd(sm_counts + key_idx_block, find_number);
    }
    bool gt_reserve = (group_base + find_number) > RESERVE;
    int gt_vote = g.ballot(gt_reserve);
    K* key_ptr = sm_keys_ptr[key_idx_block];
    if (gt_vote == 0) {
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
          sm_possible_pos[SAME_BUF(i)][groupID * RESERVE + group_base] =
              key_pos;
          __pipeline_memcpy_async(
              sm_possible_keys[SAME_BUF(i)] + groupID * RESERVE + group_base,
              key_ptr + key_pos, sizeof(K));
          group_base += 1;
        } else {
          break;
        }
      } while (true);
    } else {
      K target_key = sm_target_keys[key_idx_block];
      sm_counts[key_idx_block] = 0;
      int found_vote = 0;
      bool found = false;
      do {
        int digest_idx = __ffs(find_result) - 1;
        if (digest_idx >= 0) {
          find_result &= (find_result - 1);
          int key_pos = digest_idx < 4
                            ? (rank * 4 + digest_idx)
                            : ((GROUP_SIZE + rank - 1) * 4 + digest_idx);
          K possible_key = key_ptr[key_pos];
          if (possible_key == target_key) {
            found = true;
            sm_counts[key_idx_block] = 1;
            sm_possible_pos[SAME_BUF(i)][groupID * RESERVE] = key_pos;
            sm_possible_keys[SAME_BUF(i)][groupID * RESERVE] = possible_key;
          }
        }
        found_vote = g.ballot(found);
        if (found_vote) {
          break;
        }
        found_vote = digest_idx >= 0;
      } while (g.any(found_vote));
    }
    __pipeline_commit();

    /* Step3: check possible keys, and prefecth the value and score */
    if (i > 0) {
      key_idx_block -= 1;
      K target_key = sm_target_keys[key_idx_block];
      int possible_num = sm_counts[key_idx_block];
      sm_founds[key_idx_block] = 0;
      S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
      V* value_ptr = sm_values_ptr[key_idx_block];
      __pipeline_wait_prior(3);
      int key_pos;
      bool found_flag = false;
      if (rank < possible_num) {
        K possible_key =
            sm_possible_keys[DIFF_BUF(i)][groupID * RESERVE + rank];
        key_pos = sm_possible_pos[DIFF_BUF(i)][groupID * RESERVE + rank];
        if (possible_key == target_key) {
          found_flag = true;
          CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                             score_ptr + key_pos);
        }
      }
      int found_vote = g.ballot(found_flag);
      if (found_vote) {
        sm_founds[key_idx_block] = 1;
        int src_lane = __ffs(found_vote) - 1;
        int target_pos = g.shfl(key_pos, src_lane);
        V* v_src = value_ptr + target_pos * dim;
        V* v_dst = sm_vector[DIFF_BUF(i)][groupID];
        CopyValue::ldg_sts(rank, v_dst, v_src, dim);
      }
    }
    __pipeline_commit();

    /* Step4: write back value and score */
    if (i > 1) {
      key_idx_block -= 1;
      int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
      int found_flag = sm_founds[key_idx_block];
      V* v_src = sm_vector[SAME_BUF(i)][groupID];
      V* v_dst = values + key_idx_grid * dim;
      __pipeline_wait_prior(3);
      if (found_flag > 0) {
        S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
        CopyValue::lds_stg(rank, v_dst, v_src, dim);
        founds[key_idx_grid] = true;
        CopyScore::stg(scores + key_idx_grid, score_);
      }
    }
  }  // End loop

  /* Pipeline emptying: step3, i = loop_num */
  {
    int key_idx_block = groupID * GROUP_SIZE + (loop_num - 1);
    K target_key = sm_target_keys[key_idx_block];
    int possible_num = sm_counts[key_idx_block];
    sm_founds[key_idx_block] = 0;
    S* score_ptr = CopyScore::get_base_ptr(sm_keys_ptr, key_idx_block);
    V* value_ptr = sm_values_ptr[key_idx_block];
    __pipeline_wait_prior(1);
    int key_pos;
    bool found_flag = false;
    if (rank < possible_num) {
      key_pos = sm_possible_pos[DIFF_BUF(loop_num)][groupID * RESERVE + rank];
      K possible_key =
          sm_possible_keys[DIFF_BUF(loop_num)][groupID * RESERVE + rank];
      if (possible_key == target_key) {
        found_flag = true;
        CopyScore::ldg_sts(sm_target_scores + key_idx_block,
                           score_ptr + key_pos);
      }
    }
    int found_vote = g.ballot(found_flag);
    if (found_vote) {
      sm_founds[key_idx_block] = 1;
      int src_lane = __ffs(found_vote) - 1;
      int target_pos = g.shfl(key_pos, src_lane);
      V* v_src = value_ptr + target_pos * dim;
      V* v_dst = sm_vector[DIFF_BUF(loop_num)][groupID];
      CopyValue::ldg_sts(rank, v_dst, v_src, dim);
    }
  }
  __pipeline_commit();

  /* Pipeline emptying: step4, i = loop_num */
  if (loop_num > 1) {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 2;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_dst = values + key_idx_grid * dim;
    V* v_src = sm_vector[SAME_BUF(loop_num)][groupID];
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(1);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }

  /* Pipeline emptying: step4, i = loop_num + 1 */
  {
    int key_idx_block = groupID * GROUP_SIZE + loop_num - 1;
    int key_idx_grid = blockIdx.x * blockDim.x + key_idx_block;
    V* v_dst = values + key_idx_grid * dim;
    V* v_src = sm_vector[SAME_BUF(loop_num + 1)][groupID];
    int found_flag = sm_founds[key_idx_block];
    __pipeline_wait_prior(0);
    if (found_flag > 0) {
      S score_ = CopyScore::lgs(sm_target_scores + key_idx_block);
      CopyValue::lds_stg(rank, v_dst, v_src, dim);
      founds[key_idx_grid] = true;
      CopyScore::stg(scores + key_idx_grid, score_);
    }
  }
}  // End function

template <typename K, typename V, typename S>
struct SelectLookupKernelWithIOPipeline {
  static void execute_kernel(const Bucket<K, V, S>* __restrict buckets,
                             const size_t buckets_num, const int dim_,
                             cudaStream_t& stream, const size_t& n,
                             const K* __restrict keys, V* __restrict values,
                             S* __restrict scores, bool* __restrict founds) {
    constexpr int BUCKET_SIZE = 128;
    constexpr int BLOCK_SIZE = 128;

    const uint32_t value_size = dim_ * sizeof(V);
    const uint32_t dim = value_size / sizeof(float);

    // Using 32 threads to deal with one key
    if (value_size > (128 * sizeof(float))) {
      constexpr int GROUP_SIZE = 32;
      if (scores == nullptr) {
        using CopyScore = CopyScoreEmpty<S, K, BUCKET_SIZE>;
        if (value_size % sizeof(float4) == 0) {
          using CopyValue = CopyValueTwoGroup<float, float4, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        } else if (value_size % sizeof(float2) == 0) {
          using CopyValue = CopyValueMultipleGroup<float, float2, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        } else {
          assert(value_size % sizeof(float) == 0);
          using CopyValue = CopyValueMultipleGroup<float, float, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        }
      } else {
        using CopyScore = CopyScoreByPassCache<S, K, BUCKET_SIZE>;
        if (value_size % sizeof(float4) == 0) {
          using CopyValue = CopyValueTwoGroup<float, float4, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        } else if (value_size % sizeof(float2) == 0) {
          using CopyValue = CopyValueMultipleGroup<float, float2, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        } else {
          assert(value_size % sizeof(float) == 0);
          using CopyValue = CopyValueMultipleGroup<float, float, GROUP_SIZE>;
          lookup_kernel_with_io_pipeline_v1<K, float, S, CopyScore, CopyValue>
              <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                  buckets, buckets_num, dim, keys, values, scores, founds, n);
        }
      }
    } else {
      // Using 16 threads to deal with one key
      constexpr int GROUP_SIZE = 16;
      if (scores == nullptr) {
        using CopyScore = CopyScoreEmpty<S, K, BUCKET_SIZE>;
        if (value_size % sizeof(float4) == 0) {
          if (value_size > (GROUP_SIZE * sizeof(float4))) {
            using CopyValue = CopyValueTwoGroup<float, float4, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float4, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        } else if (value_size % sizeof(float2) == 0) {
          if (value_size > (2 * GROUP_SIZE * sizeof(float2))) {
            using CopyValue = CopyValueMultipleGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else if (value_size > (GROUP_SIZE * sizeof(float2))) {
            using CopyValue = CopyValueTwoGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        } else {
          assert(value_size % sizeof(float) == 0);
          if (value_size > (2 * GROUP_SIZE * sizeof(float))) {
            using CopyValue = CopyValueMultipleGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else if (value_size > (GROUP_SIZE * sizeof(float))) {
            using CopyValue = CopyValueTwoGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        }
      } else {
        using CopyScore = CopyScoreByPassCache<S, K, BUCKET_SIZE>;
        if (value_size % sizeof(float4) == 0) {
          if (value_size > (GROUP_SIZE * sizeof(float4))) {
            using CopyValue = CopyValueTwoGroup<float, float4, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float4, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        } else if (value_size % sizeof(float2) == 0) {
          if (value_size > (2 * GROUP_SIZE * sizeof(float2))) {
            using CopyValue = CopyValueMultipleGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else if (value_size > (GROUP_SIZE * sizeof(float2))) {
            using CopyValue = CopyValueTwoGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float2, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        } else {
          assert(value_size % sizeof(float) == 0);
          if (value_size > (2 * GROUP_SIZE * sizeof(float))) {
            using CopyValue = CopyValueMultipleGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else if (value_size > (GROUP_SIZE * sizeof(float))) {
            using CopyValue = CopyValueTwoGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          } else {
            using CopyValue = CopyValueOneGroup<float, float, GROUP_SIZE>;
            lookup_kernel_with_io_pipeline_v2<K, float, S, CopyScore, CopyValue>
                <<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                    buckets, buckets_num, dim, keys, values, scores, founds, n);
          }
        }
      }
    }
  }  // End function
};

}  // namespace merlin
}  // namespace nv