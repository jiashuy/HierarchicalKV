/*
  no atomic
  no function call
  use __ballot_sync instead of ballot : no help
  prefetch meta to register
  align to cacheline size to avoid uncoalesed memory access : don't fit with all case
  use the specail key instead of lock the bucket
  
  unroll the code for [iteration] times
*/

// threads in a warp cooperatively deal with one key at the same time
// a warp deals with [iteration] keys, unroll the loop to hide lantency
// more iterations bring more register consumption, greatly decrease the occupancy:
//  use registers(local memory) to store intermediate results
template <class K, class V, class M,
          uint32_t iteration = 4,
          uint32_t block_dim = 128,
          uint32_t TILE_SIZE = 32,
          uint32_t warp_num = (block_dim / TILE_SIZE)>
__forceinline__ __device__ void lookup_kernel_with_io_core_v3(
    const Table<K, V, M>* __restrict table, const K* __restrict keys,
    V* __restrict values, M* __restrict metas, bool* __restrict found,
    size_t N) {
  Bucket<K, V, M>* buckets = table->buckets;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t buckets_num = table->buckets_num;
  const size_t dim = table->dim;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  // use __ballot_sync instead of ballot 
  // int rank = threadIdx.x & 0x1f;

  int rank = g.thread_rank();
  K find_key[iteration];
  V* find_value[iteration];

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE * iteration;

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      find_key[i] = keys[key_idx + i];
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      find_value[i] = values + (key_idx + i) * dim;
    }

    uint32_t hashed_key[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      hashed_key[i] = Murmur3HashDevice(find_key[i]);
    }

    size_t global_idx[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      global_idx[i] = hashed_key[i] & (buckets_num * bucket_max_size - 1);
    }

    size_t bkt_idx[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      bkt_idx[i] = global_idx[i] / bucket_max_size;
    }

    Bucket<K, V, M>* bucket[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      bucket[i] = buckets + bkt_idx[i];
    }

    K* key_base[iteration];
    M* meta_base[iteration];
    V* value_resue[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      key_base[i] = reinterpret_cast<K*>(bucket[i]->keys);
    }
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      meta_base[i] = reinterpret_cast<M*>(bucket[i]->metas);
    }
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      value_resue[i] = reinterpret_cast<V*>(bucket[i]->vectors);
    }

    size_t start_idx[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      start_idx[i] = global_idx[i] % bucket_max_size;
    }

    // // align to cacheline size
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      start_idx[i] = start_idx[i] - (start_idx[i] % 16);
    }

    bool predicate[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      predicate[i] = true;
    }

    int key_pos[iteration];
    uint32_t tile_offset[iteration];
    unsigned found_vote[iteration];

    for (int offset = 0; offset < bucket_max_size; offset += TILE_SIZE) {

      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          tile_offset[i] = offset;
        }
      }

      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          key_pos[i] =
              (start_idx[i] + offset + rank) & (bucket_max_size - 1);
        }
      }

      K* key_addr[iteration];
      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          key_addr[i] = reinterpret_cast<K*>(key_base[i] + key_pos[i]);
        }
      }

      K current_key[iteration];
      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          current_key[i] = *(key_addr[i]);
        }
      }

      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          found_vote[i] = g.ballot(find_key[i] == current_key[i]);
        }
      }

      #pragma unroll
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          if (found_vote[i]) {
            predicate[i] = false;
          }
        }
      }
    }

    int src_lane[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        src_lane[i] = __ffs(found_vote[i]) - 1;
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        key_pos[i] = (start_idx[i] + tile_offset[i] + src_lane[i]) & (bucket_max_size - 1);
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        value_resue[i] = value_resue[i] + key_pos[i] * dim;
      }
    }

    M* meta_addr[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        if (metas != nullptr) {
          meta_addr[i] = reinterpret_cast<M*>(meta_base[i] + key_pos[i]);
        }
      }
    }

    M meta_prefetch[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        if (metas != nullptr) {
          meta_prefetch[i] = *(meta_addr[i]);
        }
      }
    }

    bool occupy_res[iteration];
    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        occupy_res[i] = reinterpret_cast<AtomicKey<K>*>(key_base[i])[key_pos[i]].compare_exchange_strong(
                      find_key[i], static_cast<K>(OCCUPY_KEY), cuda::std::memory_order_relaxed);
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        if (!occupy_res[i]) {
          found_vote[i] = false;
        }
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        for (auto j = rank; j < dim; j += TILE_SIZE) {
          find_value[i][j] = value_resue[i][j];
        }
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        auto expected_key = static_cast<K>(OCCUPY_KEY);
        reinterpret_cast<AtomicKey<K>*>(key_base[i])[key_pos[i]].compare_exchange_strong(
                      expected_key, find_key[i], cuda::std::memory_order_relaxed);
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        *(found + key_idx + i) = true;
      }
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[i]) {
        if (metas != nullptr) {
          metas[key_idx + i] = meta_prefetch[i];
        }
      }
    }
  }
}