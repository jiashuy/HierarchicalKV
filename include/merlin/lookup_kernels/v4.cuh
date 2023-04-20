/*
  no atomic
  no function call
  use __ballot_sync instead of ballot : no help
  prefetch meta to register
  align to cacheline size to avoid uncoalesed memory access : don't fit with all case
  use the specail key instead of lock the bucket
  
  unroll the code for [iteration] times

  intermediate results are placed in shared memory
*/

// threads in a warp cooperatively deal with one key at the same time
// a warp deals with [iteration] keys, unroll the loop to hide lantency
// more iterations bring more register consumption, greatly decrease the occupancy:
//  use shared memory to store intermediate results
template <class K, class V, class M,
          uint32_t iteration = 8,
          uint32_t block_dim = 128,
          uint32_t TILE_SIZE = 32,
          uint32_t warp_num = (block_dim / TILE_SIZE)>
__forceinline__ __device__ void lookup_kernel_with_io_core_v4(
    const Table<K, V, M>* __restrict table, const K* __restrict keys,
    V* __restrict values, M* __restrict metas, bool* __restrict found,
    size_t N) {

  // avoid contention between warps
  __shared__ Bucket<K, V, M>* buckets[warp_num];
  __shared__ size_t bucket_max_size[warp_num];
  __shared__ size_t buckets_num[warp_num];
  __shared__ size_t dim[warp_num];

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  // int rank = threadIdx.x & 0x1f;
  int rank = g.thread_rank();
  int warpId = threadIdx.x >> 5;

  //if (rank == 0) {
    dim[warpId] = table->dim;
    buckets[warpId] = table->buckets;
    bucket_max_size[warpId] = table->bucket_max_size;
    buckets_num[warpId] = table->buckets_num;
  //}

  __shared__ K find_key[warp_num][iteration];
  __shared__ V* find_value[warp_num][iteration];

  __shared__ K* key_base[warp_num][iteration];
  __shared__ M* meta_base[warp_num][iteration];
  __shared__ V* value_resue[warp_num][iteration];

  __shared__ size_t start_idx[warp_num][iteration];
  __shared__ uint32_t tile_offset[warp_num][iteration];
  __shared__ M shared_meta[warp_num][iteration];
  __shared__ K shared_key[warp_num][iteration * TILE_SIZE];

  __shared__ unsigned found_vote[warp_num][iteration];
  __shared__ int key_pos_shared[warp_num][iteration];

  // if one key is found, then stop to traverse the bucket
  bool predicate[iteration];

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE * iteration;

    if (rank < iteration) {
      auto find_key_ = keys[key_idx + rank];
      find_key[warpId][rank] = find_key_;
      find_value[warpId][rank] = values + (key_idx + rank) * dim[warpId];

      uint32_t hashed_key = Murmur3HashDevice(find_key_);
      size_t global_idx = hashed_key & (buckets_num[warpId] * bucket_max_size[warpId] - 1);
      auto start_idx_ = global_idx % bucket_max_size[warpId];
      size_t bkt_idx = global_idx / bucket_max_size[warpId];
      Bucket<K, V, M>* bucket = buckets[warpId] + bkt_idx;
      // align to cacheline size
      start_idx[warpId][rank] = start_idx_ - (start_idx_ % 16);

      key_base[warpId][rank] = reinterpret_cast<K*>(bucket->keys);
      meta_base[warpId][rank] = reinterpret_cast<M*>(bucket->metas);
      value_resue[warpId][rank] = reinterpret_cast<V*>(bucket->vectors);
    }

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      predicate[i] = true;
    }
      

    __syncwarp();

    for (int offset = 0; offset < bucket_max_size[warpId]; offset += TILE_SIZE) {
      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          if (rank == 0) {
            tile_offset[warpId][i] = offset;
          }
          int key_pos = (start_idx[warpId][i] + offset + rank) & (bucket_max_size[warpId] - 1);
          K* key_addr = reinterpret_cast<K*>(key_base[warpId][i] + key_pos);
          shared_key[warpId][i * TILE_SIZE + rank] = *key_addr;
        }
      }

      for (int i = 0; i < iteration; i++) {
        if (predicate[i]) {
          auto current_key = shared_key[warpId][i * TILE_SIZE + rank];
          unsigned found_vote_ = g.ballot(find_key[warpId][i] == current_key);
          // unsigned found_vote_ = __ballot_sync(0xffffffff, find_key[warpId][i] == current_key);
          if (rank == 0)
            found_vote[warpId][i] = found_vote_;
          if (found_vote_) {// || g.any(current_key == static_cast<K>(EMPTY_KEY))) {
          // if (found_vote_ || __any_sync(0xffffffff, current_key == static_cast<K>(EMPTY_KEY))) {
            predicate[i] = false;
          }
        }
      }
    }
    __syncwarp();

    if (rank < iteration) {
      if (found_vote[warpId][rank]) {
        const int src_lane = __ffs(found_vote[warpId][rank]) - 1;
        int key_pos_ = (start_idx[warpId][rank] + tile_offset[warpId][rank] + src_lane) & (bucket_max_size[warpId] - 1);
        value_resue[warpId][rank] = value_resue[warpId][rank] + key_pos_ * dim[warpId];
        key_pos_shared[warpId][rank] = key_pos_;
        if (metas != nullptr) {
          auto meta_addr = reinterpret_cast<M*>(meta_base[warpId][rank] + key_pos_);
          shared_meta[warpId][rank] = *meta_addr;
        }
      }
    }

    __syncwarp();

    K expected_key;
    int key_pos_;
    if (rank < iteration) {
      if (found_vote[warpId][rank]) {
        key_pos_ = key_pos_shared[warpId][rank];
        expected_key = find_key[warpId][rank];
        bool res = reinterpret_cast<AtomicKey<K>*>(key_base[warpId][rank])[key_pos_].compare_exchange_strong(
                      expected_key, static_cast<K>(OCCUPY_KEY), cuda::std::memory_order_relaxed);
        if (!res) {
          found_vote[warpId][rank] = false;
        }
      }
    }
    __syncwarp();

    #pragma unroll
    for (int i = 0; i < iteration; i++) {
      if (found_vote[warpId][i]) {
        for (auto j = rank; j < dim[warpId]; j += TILE_SIZE) {
          find_value[warpId][i][j] = value_resue[warpId][i][j];
        }
      }
    }
    if (rank < iteration) {
      if (found_vote[warpId][rank]) {
        auto expected_new = static_cast<K>(OCCUPY_KEY);
        reinterpret_cast<AtomicKey<K>*>(key_base[warpId][rank])[key_pos_].compare_exchange_strong(
                      expected_new, expected_key, cuda::std::memory_order_relaxed);
        *(found + key_idx + rank) = true;
        if (metas != nullptr) {
          metas[key_idx + rank] = shared_meta[warpId][rank];
        }
      }
    }
  }
}