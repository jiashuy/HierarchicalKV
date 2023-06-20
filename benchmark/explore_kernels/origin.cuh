#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;
/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel_with_io_origin(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V* __restrict values, S* __restrict scores, bool* __restrict found,
    size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, S>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    V* find_value = values + key_idx * dim;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim, find_value,
                                dim);
      if (rank == src_lane) {
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel_with_io_origin_probing(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V** __restrict values_addr, S* __restrict scores, bool* __restrict found,
    size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, S>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    // V* find_value = values + key_idx * dim;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, S>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, S, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      values_addr[key_idx] = bucket->vectors + key_pos * dim;
      // copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim, find_value,
      //                           dim);
      if (rank == src_lane) {
        if (scores != nullptr) {
          *(scores + key_idx) =
              bucket->scores(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}

template <typename V,
  int32_t GROUP_SIZE = 32>
__global__ void lookup_kernel_with_io_core_origin_copying(
  V*  values, 
  V**  values_addr, 
  bool* founds, 
  int dim, 
  size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int rank = tid % GROUP_SIZE;
  int groupID = tid / GROUP_SIZE;
  const int GROUP_NUM = (blockDim.x * gridDim.x) / GROUP_SIZE; 

  for (int v_idx = groupID; v_idx < n; v_idx += GROUP_NUM) {
    if (founds[v_idx]) {
      auto v_src = values_addr[v_idx];
      auto v_dst = values + v_idx * dim;
      for (auto j = rank; j < dim; j += GROUP_SIZE) {
        v_dst[j] = v_src[j];
      }
    }
  }
}