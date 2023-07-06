#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;

// 1 thread probing using digest, group reduction and copying
template <class K, class V, class S, uint32_t GROUP_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core_beta_v1(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores,
    K* __restrict evicted_keys, V* __restrict evicted_values,
    S* __restrict evicted_scores, size_t N) {
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
    uint8_t probing_digest;
    uint8_t digest = get_digest<K>(insert_key);
    bool result = false;

    for (int tile_offset = 0; tile_offset < bucket_max_size; tile_offset++) {
      key_pos = (start_idx + tile_offset) % bucket_max_size;
      probing_digest = (bucket->digests(key_pos))[0];
      if (digest == probing_digest) {
        expected_key = insert_key;
        current_key = bucket->keys(key_pos);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
        if (result) {
          update_score(bucket, key_pos, scores, key_idx);
          bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
          occupy_result = OccupyResult::DUPLICATE;
          goto REDUCE;
        }
      }
      if (probing_digest == empty_digest<K>()) {
        expected_key = static_cast<K>(EMPTY_KEY);
        current_key = bucket->keys(key_pos);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
        if (result) {
          update_score(bucket, key_pos, scores, key_idx);
          bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
          occupy_result = OccupyResult::OCCUPIED_EMPTY;
          atomicAdd(&(buckets_size[bkt_idx]), 1);
          goto REDUCE;
        }        
      }
    }
    occupy_result = OccupyResult::CONTINUE; 
  }
REDUCE:
  g.sync();
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

  for (int i = 0; i < GROUP_SIZE; i++) {
    auto occupy_result_cur = g.shfl(occupy_result, i);
    if (occupy_result_cur == OccupyResult::ILLEGAL) continue;

    auto insert_value_cur = g.shfl(insert_value, i);
    auto key_idx_cur = g.shfl(key_idx, i);
    auto key_pos_cur = g.shfl(key_pos, i);
    auto bucket_cur = g.shfl(bucket, i);

    if (occupy_result_cur == OccupyResult::REFUSED) {
      copy_vector<V, GROUP_SIZE>(g, insert_value_cur, evicted_values + key_idx_cur * dim,
                                dim);
      continue;
    }

    if (occupy_result_cur == OccupyResult::EVICT) {
      copy_vector<V, GROUP_SIZE>(g, bucket_cur->vectors + key_pos_cur * dim,
                                evicted_values + key_idx_cur * dim, dim);
    }

    copy_vector<V, GROUP_SIZE>(g, insert_value_cur, bucket_cur->vectors + key_pos_cur * dim,
                              dim);
    if (g.thread_rank() == i) {
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}