#pragma once
#include "../../tests/test_util.cuh"

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void upsert_kernel_with_io_core_evict(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores, size_t N, size_t* evit_number) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

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

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == 0) {
        atomicAdd(evit_number, 1);
      }
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_score(bucket, key_pos, scores, key_idx);
      bucket->digests_16[key_pos] = get_digest_16<K>(insert_key);
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <class K, class V, class S, uint32_t TILE_SIZE = 4>
__global__ void upsert_kernel_with_io_core_evict_max_score(
    const Table<K, V, S>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const S* __restrict scores, size_t N, S* max_score) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

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

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == 0) {
        atomicMax(reinterpret_cast<unsigned long long int*>(max_score), 
          reinterpret_cast<unsigned long long int*>(bucket->scores(key_pos))[0]);
      }
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_score(bucket, key_pos, scores, key_idx);
      bucket->digests_16[key_pos] = get_digest_16<K>(insert_key);
      bucket->digests(key_pos)[0] = get_digest<K>(insert_key);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

float test_when_evict_occur(TestDescriptor& td, size_t& evict_number, bool silence = true) {

  const uint64_t key_num_per_op = td.key_num_per_op;
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaSetDevice(0));
    init = true;
  }
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free / (1 << 30) < td.HBM4Values) {
    if (!silence) 
      std::cout << "HBM is not enough!\n";
    return -1.0f;
  }

  uint64_t key_num_init = static_cast<uint64_t>(td.capacity * td.load_factor);
  if (key_num_init < key_num_per_op && Hit_Mode::last_insert == td.hit_mode) {
    if (!silence) 
      std::cout << "Keys for init is too few!\n";
    return -1.0f;
  }

  TableOptions options;
  options.init_capacity = td.capacity;
  options.max_capacity = td.capacity;
  options.dim = td.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.io_by_cpu = false;
  options.evict_strategy = EvictStrategy::kCustomized;
  options.max_bucket_size = td.bucket_size;
  constexpr int block_size = 128;

  std::unique_ptr<HashTable_> table = std::make_unique<HashTable_>();
  table->init(options);

  TableCore_* table_core = table->get_host_table();
  TableCore_* table_core_device = table->get_device_table();

  K* h_keys;
  S* h_scores;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;

  size_t* d_evict_number;

  CUDA_CHECK(cudaMalloc(&d_evict_number,  sizeof(size_t)));
  CUDA_CHECK(cudaMemset(d_evict_number, 0, sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  K start = 0UL;
  while (real_load_factor < 0.98f) {
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_per_op, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S),
                          cudaMemcpyHostToDevice));
    ///////////////////////////////////////////////////////////////
    if (real_load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number); 
    } else if (real_load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number); 
    } else {
      const unsigned int tile_size = 32;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number);
    }
    ///////////////////////////////////////////////////////////////
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_per_op;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaMemcpy(&evict_number, d_evict_number, sizeof(size_t),
                          cudaMemcpyDeviceToHost));
    if (evict_number != 0) 
      break;
  }

  uint32_t hmem4values =
      td.capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024UL);
  hmem4values = hmem4values < td.HBM4Values ? 0 : (hmem4values - td.HBM4Values);
  td.HMEM4Values = hmem4values;

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_evict_number));
  
  CudaCheckError();
  return real_load_factor;
}

float test_evict_number_when_bucket_full(TestDescriptor& td, size_t& evict_number, bool silence = true) {

  const uint64_t key_num_per_op = td.key_num_per_op;
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaSetDevice(0));
    init = true;
  }
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free / (1 << 30) < td.HBM4Values) {
    if (!silence) 
      std::cout << "HBM is not enough!\n";
    return -1.0f;
  }

  uint64_t key_num_init = static_cast<uint64_t>(td.capacity * td.load_factor);
  if (key_num_init < key_num_per_op && Hit_Mode::last_insert == td.hit_mode) {
    if (!silence) 
      std::cout << "Keys for init is too few!\n";
    return -1.0f;
  }

  TableOptions options;
  options.init_capacity = td.capacity;
  options.max_capacity = td.capacity;
  options.dim = td.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.io_by_cpu = false;
  options.evict_strategy = EvictStrategy::kCustomized;
  options.max_bucket_size = td.bucket_size;
  constexpr int block_size = 128;

  std::unique_ptr<HashTable_> table = std::make_unique<HashTable_>();
  table->init(options);

  TableCore_* table_core = table->get_host_table();
  TableCore_* table_core_device = table->get_device_table();

  K* h_keys;
  S* h_scores;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;

  size_t* d_evict_number;

  CUDA_CHECK(cudaMalloc(&d_evict_number,  sizeof(size_t)));
  CUDA_CHECK(cudaMemset(d_evict_number, 0, sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  K start = 0UL;
  while (real_load_factor < 0.98f) {
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_per_op, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S),
                          cudaMemcpyHostToDevice));
    ///////////////////////////////////////////////////////////////
    if (real_load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number); 
    } else if (real_load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number); 
    } else {
      const unsigned int tile_size = 32;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_evict_number);
    }
    ///////////////////////////////////////////////////////////////
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_per_op;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  // std::cout << start << std::endl;

  CUDA_CHECK(cudaMemcpy(&evict_number, d_evict_number, sizeof(size_t),
                        cudaMemcpyDeviceToHost));

  uint32_t hmem4values =
      td.capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024UL);
  hmem4values = hmem4values < td.HBM4Values ? 0 : (hmem4values - td.HBM4Values);
  td.HMEM4Values = hmem4values;

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_evict_number));
  
  CudaCheckError();
  return real_load_factor;
}

float test_max_score_evicted_per_interval(TestDescriptor& td, bool silence = true) {

  const uint64_t key_num_per_op = td.key_num_per_op;
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaSetDevice(0));
    init = true;
  }
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free / (1 << 30) < td.HBM4Values) {
    if (!silence) 
      std::cout << "HBM is not enough!\n";
    return -1.0f;
  }

  uint64_t key_num_init = static_cast<uint64_t>(td.capacity * td.load_factor);
  if (key_num_init < key_num_per_op && Hit_Mode::last_insert == td.hit_mode) {
    if (!silence) 
      std::cout << "Keys for init is too few!\n";
    return -1.0f;
  }

  TableOptions options;
  options.init_capacity = td.capacity;
  options.max_capacity = td.capacity;
  options.dim = td.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.io_by_cpu = false;
  options.evict_strategy = EvictStrategy::kCustomized;
  options.max_bucket_size = td.bucket_size;
  constexpr int block_size = 128;

  std::unique_ptr<HashTable_> table = std::make_unique<HashTable_>();
  table->init(options);

  TableCore_* table_core = table->get_host_table();
  TableCore_* table_core_device = table->get_device_table();

  K* h_keys;
  S* h_scores;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_scores, key_num_per_op * sizeof(S)));

  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;

  S* d_max_score;

  CUDA_CHECK(cudaMalloc(&d_max_score,  sizeof(S)));
  CUDA_CHECK(cudaMemset(d_max_score, 0, sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, key_num_per_op * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  K start = 0UL;
  S h_max_score = 0;
  float target_load_factor = 0.02f;
  while (real_load_factor < 0.999f) {

    if ((real_load_factor - target_load_factor > 0) 
      && (real_load_factor - target_load_factor < EPSILON)) {
      std::cout << "Load factor: " << target_load_factor << "\t\t"
                << "Max score: " << h_max_score << "\n";
      target_load_factor += 0.03f;
      CUDA_CHECK(cudaMemset(d_max_score, 0, sizeof(S)));
    }

    create_continuous_keys<K, S>(h_keys, h_scores, key_num_per_op, start);
    for (int i = 0; i < key_num_per_op; i++) {
      h_scores[i] = h_keys[i];
    }
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, key_num_per_op * sizeof(S),
                          cudaMemcpyHostToDevice));
    ///////////////////////////////////////////////////////////////
    if (real_load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict_max_score<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_max_score); 
    } else if (real_load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict_max_score<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_max_score); 
    } else {
      const unsigned int tile_size = 32;
      const size_t N = key_num_per_op * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core_evict_max_score<K, V, S, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
            table_core_device, options.max_bucket_size, table_core->buckets_num, 
            options.dim, d_keys, d_vectors, d_scores, N, d_max_score);
    }
    ///////////////////////////////////////////////////////////////
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_per_op;
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaMemcpy(&h_max_score, d_max_score, sizeof(size_t),
                          cudaMemcpyDeviceToHost));
  }

  uint32_t hmem4values =
      td.capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024UL);
  hmem4values = hmem4values < td.HBM4Values ? 0 : (hmem4values - td.HBM4Values);
  td.HMEM4Values = hmem4values;

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_scores));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_max_score));
  
  CudaCheckError();
  return real_load_factor;
}

#define ASSERT_EQ(x1, x2)             \
{                                     \
  if ((x1) != (x2)) {                 \
    std::cout << "Unequal!\n";        \
  }                                   \
}                                          

#define ASSERT_GE(x1, x2)            \
{                                    \
  if ((x1) < (x2)) {                 \
    std::cout << "Less Than!\n";     \
  }                                  \
}

void test_evict_strategy_customized_correct_rate(TestDescriptor& td) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 64;
  const uint64_t MAX_BUCKET_SIZE = td.bucket_size;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr int DIM = 4;
  ASSERT_EQ(DIM, td.dim);
  // float expected_correct_rate = 0.964;
  const int rounds = 3;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = td.dim;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;

  K* h_keys_base = test_util::HostBuffer<K>(BATCH_SIZE).ptr();
  S* h_scores_base = test_util::HostBuffer<S>(BATCH_SIZE).ptr();
  V* h_vectors_base = test_util::HostBuffer<V>(BATCH_SIZE * options.dim).ptr();

  K* h_keys_temp = test_util::HostBuffer<K>(MAX_CAPACITY).ptr();
  S* h_scores_temp = test_util::HostBuffer<S>(MAX_CAPACITY).ptr();
  V* h_vectors_temp =
      test_util::HostBuffer<V>(MAX_CAPACITY * options.dim).ptr();

  K* d_keys_temp;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores_temp, MAX_CAPACITY * sizeof(S)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t global_start_key = 100000;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<HashTable_> table = std::make_unique<HashTable_>();
    table->init(options);
    size_t start_key = global_start_key;

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    for (int r = 0; r < rounds; r++) {
      size_t expected_min_key = global_start_key + INIT_CAPACITY * r;
      size_t expected_max_key = global_start_key + INIT_CAPACITY * (r + 1) - 1;
      // size_t expected_table_size =
      //     (r == 0) ? size_t(expected_correct_rate * INIT_CAPACITY)
      //              : INIT_CAPACITY;

      for (int s = 0; s < STEPS; s++) {
        test_util::create_continuous_keys<K, S, V, DIM>(
            h_keys_base, h_scores_base, h_vectors_base, BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;

        CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base, BATCH_SIZE * sizeof(K),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scores_temp, h_scores_base,
                              BATCH_SIZE * sizeof(S), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base,
                              BATCH_SIZE * sizeof(V) * options.dim,
                              cudaMemcpyHostToDevice));
        table->insert_or_assign(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                                d_scores_temp, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());

      size_t dump_counter = table->export_batch(
          MAX_CAPACITY, 0, d_keys_temp, d_vectors_temp, d_scores_temp, stream);

      CUDA_CHECK(cudaMemcpy(h_keys_temp, d_keys_temp, MAX_CAPACITY * sizeof(K),
                            cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_scores_temp, d_scores_temp,
                            MAX_CAPACITY * sizeof(S), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp, d_vectors_temp,
                            MAX_CAPACITY * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      ASSERT_EQ(total_size, dump_counter);
      size_t bigger_score_counter = 0;
      K max_key = 0;

      for (int i = 0; i < dump_counter; i++) {
        ASSERT_EQ(h_keys_temp[i], h_scores_temp[i]);
        max_key = std::max(max_key, h_keys_temp[i]);
        if (h_scores_temp[i] >= expected_min_key) bigger_score_counter++;
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }

      float correct_rate = (bigger_score_counter * 1.0) / MAX_CAPACITY;
      std::cout << std::setprecision(3) << "[Round " << r << "]"
                << "correct_rate=" << correct_rate << std::endl;
      ASSERT_GE(max_key, expected_max_key);
      // ASSERT_GE(correct_rate, expected_correct_rate);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_scores_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}