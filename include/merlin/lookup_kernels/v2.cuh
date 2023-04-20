/*
  no atomic
  no function call
  use __ballot_sync instead of ballot : no help
  prefetch meta to register
  align to cacheline size to avoid uncoalesed memory access : don't fit with all case
  use the specail key instead of lock the bucket
*/

// this is a temp version used to improve the performance step by step
//  to reduce unknown error
template <class K, class V, class M, uint32_t TILE_SIZE = 32>
__forceinline__ __device__ void lookup_kernel_with_io_core_v2(
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

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    K find_key = keys[key_idx];
    V* find_value = values + key_idx * dim;

    int key_pos = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;
    uint32_t tile_offset = 0;

    uint32_t hashed_key = Murmur3HashDevice(find_key);
    size_t global_idx = hashed_key & (buckets_num * bucket_max_size - 1);
    bkt_idx = global_idx / bucket_max_size;
    start_idx = global_idx % bucket_max_size;
    Bucket<K, V, M>* bucket = buckets + bkt_idx;

    unsigned found_vote;

    // // align to cacheline size
    start_idx = start_idx - (start_idx % 16);

    for (tile_offset = 0; tile_offset < bucket_max_size;
        tile_offset += TILE_SIZE) {

      key_pos =
          (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      auto key_addr = reinterpret_cast<K*>(bucket->keys + key_pos);
      auto const current_key = *key_addr;

      // use __ballot_sync instead of ballot 
      // found_vote = __ballot_sync(0xffffffff, find_key == current_key);
      // if (found_vote) {// || __any_sync(0xffffffff, current_key == static_cast<K>(EMPTY_KEY))) {
      //   break;
      // }

      found_vote = g.ballot(find_key == current_key);
      if (found_vote) {// || g.any(current_key == static_cast<K>(EMPTY_KEY))) {
        break;
      }
    }

    if (found_vote) {
      const int src_lane = __ffs(found_vote) - 1;
      key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
      const V* src = bucket->vectors + key_pos * dim;

      // prefetch meta to register
      M metas_reg;
      if (rank == 0 && metas != nullptr) {
        auto meta_addr = reinterpret_cast<M*>(bucket->metas + key_pos);
        metas_reg = *meta_addr;
      }

      // use the specail key instead of lock the bucket
      bool res = bucket->keys[key_pos].compare_exchange_strong(
                  find_key, static_cast<K>(OCCUPY_KEY), cuda::std::memory_order_relaxed);
      if (res) {
        for (auto i = rank; i < dim; i += TILE_SIZE) {
          find_value[i] = src[i];
        }
        auto expected_key = static_cast<K>(OCCUPY_KEY);
        bucket->keys[key_pos].compare_exchange_strong(
          expected_key, find_key, cuda::std::memory_order_relaxed);        
      } else {
        continue;
      }

      // lock<Mutex, TILE_SIZE, true>(g, table->locks[bkt_idx]);
      // for (auto i = rank; i < dim; i += TILE_SIZE) {
      //   find_value[i] = src[i];
      // }
      // unlock<Mutex, TILE_SIZE, true>(g, table->locks[bkt_idx]);

      if (rank == 0) {
        if (metas != nullptr) {
          // prefetch meta to register
          *(metas + key_idx) = metas_reg;

          // auto meta_addr = reinterpret_cast<M*>(bucket->metas + key_pos);
          // *(metas + key_idx) = *meta_addr;
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}