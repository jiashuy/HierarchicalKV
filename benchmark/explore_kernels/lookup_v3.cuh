#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;

// 8 bits digests
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v3_kernel1(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n) {
  
  constexpr int bucket_max_size = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);


  const uint8_t target_digest_ = static_cast<uint8_t>(hashed_key >> 32);
  const uint32_t target_digest = static_cast<uint32_t>(target_digest_);
  uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);

  size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  size_t bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx % bucket_max_size;

  // vectorize(128 bits) needs aligned to 16B = 8 bits x 16
  start_idx = start_idx - (start_idx % 16);

  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  uint8_t* digests_addr = reinterpret_cast<uint8_t*>(keys_addr) - bucket_max_size;
  S* scores_addr = reinterpret_cast<S*>(keys_addr + bucket_max_size);
  V* values_addr_bucket = bucket->vectors;
  int key_pos = -1;
  uint32_t tile_offset = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size; tile_offset += 16) {

    key_pos =
        (start_idx + tile_offset) & (bucket_max_size - 1);

    uint8_t* digests_ptr = digests_addr + key_pos;
    uint4 tags_ = (reinterpret_cast<uint4*>(digests_ptr))[0];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      int find_result = __vcmpeq4(probe_tags, target_digests);
      for (int j = 0; j < 4; j ++) {
        if ((find_result & 0x01) != 0) {
          int tmp = i * 4 + j;
          int possible_pos = key_pos + tmp;
          K possible_key = keys_addr[possible_pos];
          if (find_key == possible_key) {
            values_addr[key_idx] = values_addr_bucket + possible_pos * dim;
            scores[key_idx] = scores_addr[possible_pos];
            founds[key_idx] = true;
            return;
          }
        }
        find_result >>= 8;
      }
    }
  }
}

// 16 bits digests
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v3_kernel2(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n) {
  
  constexpr int bucket_max_size = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);


  const uint16_t target_digest = static_cast<uint16_t>(hashed_key >> 32);
  
  size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  size_t bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx % bucket_max_size;

  // vectorize(128 bits) needs aligned to 16B = 16 bits x 8
  start_idx = start_idx - (start_idx % 8);

  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  uint16_t* digests_addr = bucket->digests_16;
  S* scores_addr = reinterpret_cast<S*>(keys_addr + bucket_max_size);
  V* values_addr_bucket = bucket->vectors;
  int key_pos = -1;
  uint32_t tile_offset = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size; tile_offset += 8) {

    key_pos =
        (start_idx + tile_offset) & (bucket_max_size - 1);

    uint16_t* digests_ptr = digests_addr + key_pos;
    uint4 tags_ = (reinterpret_cast<uint4*>(digests_ptr))[0];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      for (int j = 0; j < 2; j++) {
        uint16_t probe_tag = static_cast<uint16_t>(probe_tags);
        if (probe_tag == target_digest) {
          int tmp = i * 2 + j;
          int possible_pos = key_pos + tmp;
          K possible_key = keys_addr[possible_pos];
          if (find_key == possible_key) {
            values_addr[key_idx] = values_addr_bucket + possible_pos * dim;
            scores[key_idx] = scores_addr[possible_pos];
            founds[key_idx] = true;
            return;
          }
        }
        probe_tags >>= 16;
      }
    }
  }
}

// probing use uint8_t
///NOTE: don't use int8_t, the compiler will pad high bits of reg using sign
template <
  class K = uint64_t,
  class V = float,
  class S = uint64_t,
  class T = uint8_t,
  class Unit = float4,
  int UnitSize = sizeof(Unit) / sizeof(V)>
__global__ void  lookup_kernel_with_io_v3_kernel3(
  Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
  const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
  bool* __restrict founds, const size_t n) {

  constexpr int BLOCK_SIZE = 128;
  constexpr int GROUP_SIZE = 32;
  constexpr int bucket_max_size = 128;

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int tx = threadIdx.x;

  int key_idx = blockIdx.x * blockDim.x + tx;
  const K find_key = keys[key_idx];

  K hashed_key = Murmur3HashDevice(find_key);
  uint8_t target_tag = static_cast<uint8_t>(hashed_key >> 32);

  int global_idx = hashed_key % (buckets_num * bucket_max_size);
  int bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx & (bucket_max_size - 1);
  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* key_base = (K*)(bucket->keys_);
  V* value_addr = (V*)(bucket->vectors);

  auto tag_base = bucket->digests_;
  auto scores_base = (S*)(key_base + bucket_max_size);
  for (int offset = 0; offset < bucket_max_size; offset += 1) {
    int key_pos = 
        (start_idx + offset) & (bucket_max_size - 1);
    uint8_t probe_tag = tag_base[key_pos];
    if (target_tag == probe_tag) {
      K possible_key = key_base[key_pos];
      if (find_key == possible_key) {
        founds[key_idx] = true;
        values_addr[key_idx] = value_addr + key_pos * dim;
        scores[key_idx] = scores_base[key_pos];
        return;
      }      
    }
  }
} // end function

// probing using uint32_t
// cost (A100 80GB) : 0.396 ms
template <
  class K = uint64_t,
  class V = float,
  class S = uint64_t,
  class T = uint8_t,
  class Unit = float4,
  int UnitSize = sizeof(Unit) / sizeof(V)>
__global__ void  lookup_kernel_with_io_v3_kernel4(
  Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
  const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
  bool* __restrict founds, const size_t n) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int GROUP_SIZE = 32;
  constexpr int bucket_max_size = 128;

  auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
  int tx = threadIdx.x;

  int key_idx = blockIdx.x * blockDim.x + tx;
  const K find_key = keys[key_idx];

  K hashed_key = Murmur3HashDevice(find_key);
  uint32_t target_tag = static_cast<uint8_t>(hashed_key >> 32);
  uint32_t target_tags = __byte_perm(target_tag, target_tag, 0x0000);

  int global_idx = hashed_key % (buckets_num * bucket_max_size);
  int bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx & (bucket_max_size - 1);
  start_idx -= start_idx % 4;
  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* key_base = (K*)(bucket->keys_);
  V* value_addr = (V*)(bucket->vectors);

  auto scores_base = reinterpret_cast<S*>(key_base + bucket_max_size);
  auto tag_base = reinterpret_cast<uint32_t*>(bucket->digests_);

  for (int offset = 0; offset < bucket_max_size; offset += 4) {
    int key_pos = 
        (start_idx + offset) & (bucket_max_size - 1);
  
    uint32_t probe_tags = tag_base[key_pos >> 2];
    int find_result = __vcmpeq4(probe_tags, target_tags);
    for (int i = 0; i < 4; i++) {
      if ((find_result & 0x01) != 0) {
        int possible_pos = key_pos + i;
        K possible_key = key_base[possible_pos];
        if (find_key == possible_key) {
          values_addr[key_idx] = value_addr + possible_pos * dim;
          scores[key_idx] = scores_base[possible_pos];
          founds[key_idx] = true;
          return;
        }
      }
      find_result >>= 8;
    }
  }
} // end function

/////////////////////////////////////////////// collect probing times ///////////////////////

// 8 bits digests
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v3_kernel1_cellect_probing_times(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n, int* times) {
  
  constexpr int bucket_max_size = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);


  const uint8_t target_digest_ = static_cast<uint8_t>(hashed_key >> 32);
  const uint32_t target_digest = static_cast<uint32_t>(target_digest_);
  uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);

  size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  size_t bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx % bucket_max_size;

  // vectorize(128 bits) needs aligned to 16B = 8 bits x 16
  start_idx = start_idx - (start_idx % 16);

  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  uint8_t* digests_addr = reinterpret_cast<uint8_t*>(keys_addr) - bucket_max_size;
  S* scores_addr = reinterpret_cast<S*>(keys_addr + bucket_max_size);
  V* values_addr_bucket = bucket->vectors;
  int key_pos = -1;
  uint32_t tile_offset = 0;
  int conflict_times = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size; tile_offset += 16) {

    key_pos =
        (start_idx + tile_offset) & (bucket_max_size - 1);

    uint8_t* digests_ptr = digests_addr + key_pos;
    uint4 tags_ = (reinterpret_cast<uint4*>(digests_ptr))[0];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      int find_result = __vcmpeq4(probe_tags, target_digests);
      for (int j = 0; j < 4; j ++) {
        if ((find_result & 0x01) != 0) {
          conflict_times += 1;
          int tmp = i * 4 + j;
          int possible_pos = key_pos + tmp;
          K possible_key = keys_addr[possible_pos];
          if (find_key == possible_key) {
            values_addr[key_idx] = values_addr_bucket + possible_pos * dim;
            scores[key_idx] = scores_addr[possible_pos];
            founds[key_idx] = true;
            times[key_idx] = (tile_offset/16 + 1) * sizeof(uint4) / sizeof(K) + conflict_times;
            return;
          }
        }
        find_result >>= 8;
      }
    }
  }
  times[key_idx] = (tile_offset/16 + 1) * sizeof(uint4) / sizeof(K) + conflict_times;
}


// 16 bits digests
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v3_kernel2_cellect_probing_times(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n, int* times) {
  
  constexpr int bucket_max_size = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);


  const uint16_t target_digest = static_cast<uint16_t>(hashed_key >> 32);
  
  size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  size_t bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx % bucket_max_size;

  // vectorize(128 bits) needs aligned to 16B = 16 bits x 8
  start_idx = start_idx - (start_idx % 8);

  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  uint16_t* digests_addr = bucket->digests_16;
  S* scores_addr = reinterpret_cast<S*>(keys_addr + bucket_max_size);
  V* values_addr_bucket = bucket->vectors;
  int key_pos = -1;
  uint32_t tile_offset = 0;
  int conflict_times = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size; tile_offset += 8) {

    key_pos =
        (start_idx + tile_offset) & (bucket_max_size - 1);

    uint16_t* digests_ptr = digests_addr + key_pos;
    uint4 tags_ = (reinterpret_cast<uint4*>(digests_ptr))[0];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      for (int j = 0; j < 2; j++) {
        uint16_t probe_tag = static_cast<uint16_t>(probe_tags);
        if (probe_tag == target_digest) {
          conflict_times += 1;
          int tmp = i * 2 + j;
          int possible_pos = key_pos + tmp;
          K possible_key = keys_addr[possible_pos];
          if (find_key == possible_key) {
            values_addr[key_idx] = values_addr_bucket + possible_pos * dim;
            scores[key_idx] = scores_addr[possible_pos];
            founds[key_idx] = true;
            times[key_idx] = (tile_offset/8 + 1) * sizeof(uint4) / sizeof(K) + conflict_times;
            return;
          }
        }
        probe_tags >>= 16;
      }
    }
  }
  times[key_idx] = (tile_offset/8 + 1) * sizeof(uint4) / sizeof(K) + conflict_times;
}

/////////////////////////////////////////////////////////////// collect conflict times ////////////////////////
// 8 bits digests
template <class K, class V, class S>
__global__ void lookup_kernel_with_io_v3_kernel1_cellect_conflict_times(
    Bucket<K, V, S>* buckets, const size_t buckets_num, const int dim,
    const K* __restrict keys, V** __restrict values_addr, S* __restrict scores,
    bool* __restrict founds, const size_t n, int* times) {
  
  constexpr int bucket_max_size = 128;

  int key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= n) return;

  K find_key = keys[key_idx];
  K hashed_key = Murmur3HashDevice(find_key);


  const uint8_t target_digest_ = static_cast<uint8_t>(hashed_key >> 32);
  const uint32_t target_digest = static_cast<uint32_t>(target_digest_);
  uint32_t target_digests = __byte_perm(target_digest, target_digest, 0x0000);

  size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  size_t bkt_idx = global_idx / bucket_max_size;
  int start_idx = global_idx % bucket_max_size;

  // vectorize(128 bits) needs aligned to 16B = 8 bits x 16
  start_idx = start_idx - (start_idx % 16);

  Bucket<K, V, S>* bucket = buckets + bkt_idx;
  K* keys_addr = reinterpret_cast<K*>(const_cast<AtomicKey<K>*>(bucket->keys(0)));
  uint8_t* digests_addr = reinterpret_cast<uint8_t*>(keys_addr) - bucket_max_size;
  S* scores_addr = reinterpret_cast<S*>(keys_addr + bucket_max_size);
  V* values_addr_bucket = bucket->vectors;
  int key_pos = -1;
  uint32_t tile_offset = 0;
  int conflict_times = 0;

  for (tile_offset = 0; tile_offset < bucket_max_size; tile_offset += 16) {

    key_pos =
        (start_idx + tile_offset) & (bucket_max_size - 1);

    uint8_t* digests_ptr = digests_addr + key_pos;
    uint4 tags_ = (reinterpret_cast<uint4*>(digests_ptr))[0];
    uint32_t tags[4] = 
              {tags_.x, tags_.y, tags_.z, tags_.w};
    for (int i = 0; i < 4; i++) {
      uint32_t probe_tags = tags[i];
      int find_result = __vcmpeq4(probe_tags, target_digests);
      for (int j = 0; j < 4; j ++) {
        if ((find_result & 0x01) != 0) {
          conflict_times += 1;
          int tmp = i * 4 + j;
          int possible_pos = key_pos + tmp;
          K possible_key = keys_addr[possible_pos];
          if (find_key == possible_key) {
            values_addr[key_idx] = values_addr_bucket + possible_pos * dim;
            scores[key_idx] = scores_addr[possible_pos];
            founds[key_idx] = true;
            times[key_idx] = conflict_times;
            return;
          }
        }
        find_result >>= 8;
      }
    }
  }
  times[key_idx] = conflict_times;
}
