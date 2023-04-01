/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

// A100 80 GB
const float kPeakBandWidth =  1935.0f;

namespace benchmark {
enum class TimeUnit {
  Second = 0,
  MilliSecond = 3,
  MicroSecond = 6,
  NanoSecond = 9,
};

template <typename Rep>
struct Timer {
  explicit Timer(TimeUnit tu = TimeUnit::Second) : tu_(tu) {}
  void start() { startRecord = std::chrono::steady_clock::now(); }
  void end() { endRecord = std::chrono::steady_clock::now(); }
  Rep getResult() {
    auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endRecord - startRecord);
    auto pow_ =
        static_cast<int32_t>(tu_) - static_cast<int32_t>(TimeUnit::NanoSecond);
    auto factor = static_cast<Rep>(std::pow(10, pow_));
    return static_cast<Rep>(duration_.count()) * factor;
  }

 private:
  TimeUnit tu_;
  std::chrono::time_point<std::chrono::steady_clock> startRecord{};
  std::chrono::time_point<std::chrono::steady_clock> endRecord{};
};

inline void printDeviceProp(const cudaDeviceProp &prop) {
  printf("[ Device Name : %s. ]\n", prop.name);
  printf("[ Compute Capability : %d.%d ]\n", prop.major, prop.minor);
  printf("[ Total Global Memory : %ld GB ]\n", prop.totalGlobalMem/ (1024 * 1024 * 1024) );
  printf("[ Total Const Memory : %ld KB ]\n", prop.totalConstMem/ 1024 );
  printf("[ SM counts : %d. ]\n", prop.multiProcessorCount);
  printf("[ Max Block counts Per SM : %d. ]\n", prop.maxBlocksPerMultiProcessor);
  printf("[ Max GridSize[0 - 2] : %d %d %d. ]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("[ Max ThreadsDim[0 - 2] : %d %d %d. ]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("[ Max Shared Memory Per SM : %ld KB ]\n", prop.sharedMemPerMultiprocessor/1024);
  printf("[ Max Shared Memory Per Block : %ld KB ]\n", prop.sharedMemPerBlock/1024);
  printf("[ Max Registers Number Per SM : %d. ]\n", prop.regsPerMultiprocessor);
  printf("[ Max Registers Number Per Block : %d. ]\n", prop.regsPerBlock);
  printf("[ Max Threads Number Per SM : %d. ]\n", prop.maxThreadsPerMultiProcessor);
  printf("[ Max Threads Number Per Block : %d. ]\n", prop.maxThreadsPerBlock);

  printf("[ warpSize : %d. ]\n", prop.warpSize);
  printf("[ memPitch : %ld. ]\n", prop.memPitch);
  printf("[ clockRate : %d. ]\n", prop.clockRate);
  printf("[ textureAlignment : %ld. ]\n", prop.textureAlignment);
  printf("[ deviceOverlap : %d. ]\n\n", prop.deviceOverlap);

}

/*Obtain computing device information and initialize the computing device*/
inline bool init_device(int verbose, int deviceId = 0) {
  int count;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    std::cerr << "There is no device.\n";
    return false;
  }
  else {
    std::cout << "Find the device successfully.\n";
  }
  if (deviceId >= count) {
    std::cerr << "Device ID invalid";
    return false;
  }
  //set its value between 0 and n - 1 if there are n GPUS
  cudaSetDevice(deviceId);
  if (verbose == 1) {
    cudaDeviceProp prop {} ;
    cudaGetDeviceProperties(&prop, deviceId);
    printDeviceProp(prop);
  }
  return true;
}

enum class API_Select {
  find = 0,
  insert_or_assign = 1,
  find_or_insert = 2,
  assign = 3,
  insert_and_evict = 4,
};

enum class Hit_Mode {
  random = 0,
  last_insert = 1,
};

struct TestDescriptor {
  using K = uint64_t;
  using M = uint64_t;
  using V = float;

  TestDescriptor() = default;

  uint32_t bucket_size {128};
  float load_factor {0.1f};
  float hit_rate {1.0f};
  Hit_Mode hit_mode {Hit_Mode::last_insert};
  int64_t capacity {5 * 1024 * 1024UL};   
  uint64_t key_num_per_op {256 * 1024UL};
  uint32_t HBM4Values {16};   // GB
  uint32_t HMEM4Values {0};   // GB
  uint32_t dim {64};

  TimeUnit tu {TimeUnit::MilliSecond};
  float find_tm {-1.0f}; 
  float insert_or_assign_tm{-1.0f};
  float find_or_insert_tm{-1.0f};
  float assign_tm{-1.0f};
  float insert_and_evict_tm{-1.0f};

  float get_throughput(API_Select api) const {
    uint64_t mem_access = 0UL;
    float elapsed_time = 0.0f;
    switch (api) {
      case API_Select::find: {
        if (find_tm < 0) return 0.0f;
        elapsed_time = find_tm;
        mem_access = 
          ((sizeof(K) + sizeof(M) + sizeof(V) * dim) * 2 * hit_rate
            // best case, load_factor is very low
            + (sizeof(K) * 3) * (1 - hit_rate) 
            + sizeof(bool)
          ) * 1.0 * key_num_per_op;          
        break;
      }
      case API_Select::insert_or_assign: {
        if (insert_or_assign_tm < 0) return 0.0f;
        elapsed_time = insert_or_assign_tm;
        mem_access = 
          (sizeof(K) * (hit_rate * 2 + (1 - hit_rate) * 3)
            + (sizeof(M) + sizeof(V) * dim) * 2
          ) * 1.0 * key_num_per_op;
        break;
      }
      case API_Select::find_or_insert: {
        if (find_or_insert_tm < 0) return 0.0f;
        elapsed_time = find_or_insert_tm;
        mem_access = 
          (sizeof(K) * (hit_rate * 2 + (1 - hit_rate) * 3)
            + (sizeof(M) + sizeof(V) * dim) * 2
          ) * 1.0 * key_num_per_op;
        break;
      }
      case API_Select::assign :{
        if (assign_tm < 0) return 0.0f;
        elapsed_time = assign_tm;
        mem_access = 
          ((sizeof(K) + sizeof(M) + sizeof(V) * dim) * 2 * hit_rate
          + (sizeof(K) * 3) * (1 - hit_rate)
          ) * 1.0 * key_num_per_op;                                
        break;
      }
      ///TODO: more accurate
      case API_Select::insert_and_evict :{
        if (insert_and_evict_tm < 0) return 0.0f;
        elapsed_time = insert_and_evict_tm;
        mem_access = 
          (sizeof(K) * (hit_rate * 2 + (1 - hit_rate) * 3)
            + (sizeof(M) + sizeof(V) * dim) * 2
          ) * 1.0 * key_num_per_op;                             
        break;
      }
      default: {
        return -1.0f;
      }
    }
    auto pow_ =
      static_cast<int32_t>(TimeUnit::Second) - static_cast<int32_t>(tu);
    auto factor = static_cast<float>(std::pow(10, pow_));
    float time_in_seconds = elapsed_time * factor;
    float tput = 1.0 * mem_access / time_in_seconds / (1024 * 1024 * 1024.0);
    return tput / kPeakBandWidth;
  }
};


inline uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <class K, class M>
void create_random_keys(K* h_keys, M* h_metas, const int key_num_per_op) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < key_num_per_op) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_metas[i] = getTimestamp();
    i++;
  }
}

template <class K, class M>
void create_continuous_keys(K* h_keys, M* h_metas, const int key_num_per_op,
                            const K start = 0) {
  for (K i = 0; i < key_num_per_op; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = getTimestamp();
  }
}

template <typename K, typename M>
void create_keys_for_hitrate(K* h_keys, M* h_metas, const int key_num_per_op,
                     const float hitrate = 0.6f, const Hit_Mode hit_mode = Hit_Mode::last_insert,
                     const K end = 0, const bool reset = false) {
  int divide = static_cast<int>(key_num_per_op * hitrate);
  if (Hit_Mode::random == hit_mode) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    K existed_max = end == 0 ? 1 : (end - 1);
    std::uniform_int_distribution<K> distr(0, existed_max);
    
    if (existed_max < divide) {
      std::cout << "# Can not generate enough keys for hit!";
      exit(-1);
    }
    std::unordered_set<K> numbers;
    while (numbers.size() < divide) {
      numbers.insert(distr(eng));
    }
    int i = 0;
    for (auto existed_value : numbers) {
      h_keys[i] = existed_value;
      h_metas[i] = getTimestamp();
      i++;
    }
  } else {
    // else keep its original value, but update metas
    for (int i = 0; i < divide; i++) {
      h_metas[i] = getTimestamp();
    }
  }
  

  static K new_value = std::numeric_limits<K>::max();
  if (reset) {
    new_value = std::numeric_limits<K>::max();
  }
  for (int i = divide; i < key_num_per_op; i++) {
    h_keys[i] = new_value--;
    h_metas[i] = getTimestamp();
  }
}

template <typename M>
void refresh_metas(M* h_metas, const int key_num_per_op) {
  for (int i = 0; i < key_num_per_op; i++) {
    h_metas[i] = getTimestamp();
  }
}
}  // namespace benchmark
