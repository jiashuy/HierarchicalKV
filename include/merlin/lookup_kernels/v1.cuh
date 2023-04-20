template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ unsigned find_in_bucket_no_atomic(
    cg::thread_block_tile<TILE_SIZE> g,
    AtomicKey<K>* __restrict bucket_keys, const K& find_key,
    uint32_t& tile_offset, const uint32_t& start_idx,
    const size_t& bucket_max_size) {
  uint32_t key_pos = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos =
        (start_idx + tile_offset + g.thread_rank()) & (bucket_max_size - 1);
    auto key_addr = reinterpret_cast<K*>(bucket_keys + key_pos);
    auto const current_key = *key_addr;
    // auto const current_key =
    //     bucket_keys[key_pos].load(cuda::std::memory_order_relaxed);
    auto const found_vote = g.ballot(find_key == current_key);
    if (found_vote) {
      return found_vote;
    }

    if (g.any(current_key == static_cast<K>(EMPTY_KEY))) {
      return 0;
    }
  }
  return 0;
}

// no atomic version
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void lookup_kernel_with_io_core_v1(
    const Table<K, V, M>* __restrict table, const K* __restrict keys,
    V* __restrict values, M* __restrict metas, bool* __restrict found,
    size_t N) {
  Bucket<K, V, M>* buckets = table->buckets;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t buckets_num = table->buckets_num;
  const size_t dim = table->dim;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    V* find_value = values + key_idx * dim;

    int key_pos = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;
    uint32_t tile_offset = 0;

    Bucket<K, V, M>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    auto const found_vote = find_in_bucket_no_atomic<K, V, M, TILE_SIZE>(
        g, bucket->keys, find_key, tile_offset, start_idx, bucket_max_size);

    if (found_vote) {
      const int src_lane = __ffs(found_vote) - 1;
      key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
      const V* src = bucket->vectors + key_pos * dim;
      lock<Mutex, TILE_SIZE, true>(g, table->locks[bkt_idx]);
      copy_vector<V, TILE_SIZE>(g, src, find_value, dim);
      unlock<Mutex, TILE_SIZE, true>(g, table->locks[bkt_idx]);

      if (rank == 0) {
        if (metas != nullptr) {
          auto meta_addr = reinterpret_cast<M*>(bucket->metas + key_pos);
          *(metas + key_idx) = *meta_addr;
          // *(metas + key_idx) =
          //     bucket->metas[key_pos].load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}


