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

#include <cstdint>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_profiler_api.h>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

#define PROFILE
#define EPSILON 1e-3
#define PRECISION 3

const uint32_t kRepeat4Avg = 10;

inline std::string rep(int n) { return std::string(n, ' '); }

using namespace nv::merlin;
using namespace benchmark;
using namespace std;

void testOneRound(const API_Select api, TestDescriptor & td, 
                  bool sample_hit_rate = false, bool silence = true) {
  using K = uint64_t;
  using M = uint64_t;
  using V = float;
  using Table = nv::merlin::HashTable<K, float, M>;
  using TableOptions = nv::merlin::HashTableOptions;

  const uint64_t key_num_per_op = td.key_num_per_op;

  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaSetDevice(0));
    init = true;
  }
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  if (free / (1 << 30) < td.HBM4Values) {
    std::cout << "### HBM is not enough! Ignore this round test.\n";
    return;
  }

  uint64_t key_num_init = static_cast<uint64_t>(td.capacity * td.load_factor);
  if (key_num_init < key_num_per_op && Hit_Mode::last_insert == td.hit_mode) {
    std::cout << "### Init insert too few keys! Ignore this round test.\n";
    return;
  }

  TableOptions options;

  options.init_capacity = td.capacity;
  options.max_capacity = td.capacity;
  options.dim = td.dim;
  options.max_hbm_for_vectors = nv::merlin::GB(td.HBM4Values);
  options.io_by_cpu = false;
  options.evict_strategy = EvictStrategy::kCustomized;
  options.max_bucket_size = td.bucket_size;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  CUDA_CHECK(cudaMallocHost(&h_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));

  CUDA_CHECK(
      cudaMemset(h_vectors, 0, key_num_per_op * sizeof(V) * options.dim));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_vect_contrast;
  bool* d_found;

  K* d_evict_keys;
  M* d_evict_metas;

  CUDA_CHECK(cudaMalloc(&d_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, key_num_per_op * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vect_contrast, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, key_num_per_op * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_evict_keys, key_num_per_op * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_evict_metas, key_num_per_op * sizeof(M)));

  CUDA_CHECK(
      cudaMemset(d_vectors, 1, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemset(d_vect_contrast, 2, key_num_per_op * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_found, 0, key_num_per_op * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));


  const float target_load_factor = key_num_init * 1.0f / td.capacity;
  uint64_t key_num_remain = key_num_init % key_num_per_op == 0 ? 
                      key_num_per_op : key_num_init % key_num_per_op;
  int32_t loop_num_init = (key_num_init + key_num_per_op - 1) / key_num_per_op;

  // no need to get load factor
  K start = 0UL;
  for (int i = 0; i < loop_num_init; i++) {
    uint64_t key_num_cur_insert = i == loop_num_init - 1 ? 
                                    key_num_remain : key_num_per_op;

    create_continuous_keys<K, M>(h_keys, h_metas, key_num_cur_insert, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_cur_insert * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_cur_insert * sizeof(M),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign_profile(key_num_cur_insert, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_cur_insert;
  }
  if (!silence)
    std::cout << "# Loop number init insert! " << loop_num_init << std::endl;

  // read_load_factor <= target_load_factor always true, due to evict occurrence
  float real_load_factor = table->load_factor(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = 
          static_cast<int64_t>((target_load_factor - real_load_factor) * td.capacity);
    if (key_num_append <= 0) break;
    if (key_num_append > key_num_per_op) key_num_append = key_num_per_op;
    if (!silence)
      std::cout << "# Extra insert keys! " << key_num_append << std::endl;
    create_continuous_keys<K, M>(h_keys, h_metas, key_num_append, start);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_append * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_append * sizeof(M),
                          cudaMemcpyHostToDevice));
    table->insert_or_assign_profile(key_num_append, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += key_num_append;  
    real_load_factor = table->load_factor(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  if (!silence)
    std::cout << "# Load factor distance = " 
              << fixed << setprecision(PRECISION) 
              << target_load_factor - real_load_factor << std::endl;

  auto timer = Timer<double>(td.tu);
  create_keys_for_hitrate<K, M>(h_keys, h_metas, key_num_per_op, td.hit_rate, 
                                      td.hit_mode, start, true/*reset*/);
  if (sample_hit_rate) {
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                          cudaMemcpyHostToDevice));
    table->find(key_num_per_op, d_keys, d_vectors, d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_found, d_found, key_num_per_op * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    int found_num = 0;
    for (int i = 0; i < key_num_per_op; i++) {
      if (h_found[i]) {
        found_num++;
      }
    }
    float target_hit_rate = td.hit_rate;
    td.hit_rate = static_cast<float>(1.0 * found_num / key_num_per_op);
    std::cout << "# Hit rate distance = " 
              << fixed << setprecision(PRECISION) 
              << target_hit_rate - td.hit_rate << std::endl;
  }
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, key_num_per_op * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, key_num_per_op * sizeof(M),
                        cudaMemcpyHostToDevice));
  cudaProfilerStart();
  switch (api) {
    case API_Select::find: {
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors, d_found, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      td.find_tm = timer.getResult();
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(key_num_per_op, d_keys, d_vectors, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      td.insert_or_assign_tm = timer.getResult();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      td.find_or_insert_tm = timer.getResult();
      break;
    }
    case API_Select::assign :{
      timer.start();
      table->assign(key_num_per_op, d_keys, d_vect_contrast, d_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      td.assign_tm = timer.getResult();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_metas,
                              d_evict_keys, d_vect_contrast, d_evict_metas, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();
      td.insert_and_evict_tm = timer.getResult();
      break;
    }
    default: {
      std::cout << "### Unsupport API! Ignore this round test.\n";
    }
  }
  cudaProfilerStop();

  if (api == API_Select::find) {
    // quick and unsufficient verify
    bool* h_found;
    CUDA_CHECK(cudaMallocHost(&h_found, key_num_per_op * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, key_num_per_op * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    int found_num = 0;
    for (int i = 0; i < key_num_per_op; i++) {
      if (h_found[i]) found_num++;
    }
    std::cout << "found number = " << found_num << std::endl;
    CUDA_CHECK(cudaFreeHost(h_found));
  }

  uint32_t hmem4values =
      td.capacity * options.dim * sizeof(V) / (1024 * 1024 * 1024);
  hmem4values = hmem4values < td.HBM4Values ? 0 : (hmem4values - td.HBM4Values);
  td.HMEM4Values = hmem4values;
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_vect_contrast));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_evict_keys));
  CUDA_CHECK(cudaFree(d_evict_metas));

  CudaCheckError();

  return;
}

std::vector<TestDescriptor> get_average_result(
  const std::vector<std::vector<TestDescriptor>> & tds_vec,
  const std::vector<API_Select>& target_apis) {

  auto repeat_times = tds_vec.size();
  if (repeat_times == 0) return {};
  auto test_kinds = tds_vec[0].size();
  std::vector<TestDescriptor> avg_res(test_kinds);
  auto api_num = target_apis.size();
  if (api_num == 0) return avg_res;

  for (int i = 0; i < test_kinds; i++) {
    std::vector<float> sums(api_num, 0.0f);
    avg_res[i] = tds_vec[0][i];

    for (int j = 0; j < repeat_times; j++) {
      int k = 0;
      for (auto api : target_apis) {
        switch (api) {
          case API_Select::find: {
            sums[k++] += tds_vec[j][i].find_tm;
            break;
          }
          case API_Select::insert_or_assign: {
            sums[k++] += tds_vec[j][i].insert_or_assign_tm;
            break;
          }
          case API_Select::find_or_insert: {
            sums[k++] += tds_vec[j][i].find_or_insert_tm;
            break;
          }
          case API_Select::assign: {
            sums[k++] += tds_vec[j][i].assign_tm;
            break;
          }
          case API_Select::insert_and_evict: {
            sums[k++] += tds_vec[j][i].insert_and_evict_tm;
            break;
          }
          default : {
            std::cout << "[Unsupport API!] \n";
          }
        }
      }      
    }
    int k = 0;
    for (auto api : target_apis) {
      switch (api) {
        case API_Select::find: {
          avg_res[i].find_tm = sums[k++] / repeat_times;
          break;
        }
        case API_Select::insert_or_assign: {
          avg_res[i].insert_or_assign_tm = sums[k++] / repeat_times;
          break;
        }
        case API_Select::find_or_insert: {
          avg_res[i].find_or_insert_tm = sums[k++] / repeat_times;
          break;
        }
        case API_Select::assign: {
          avg_res[i].assign_tm = sums[k++] / repeat_times;
          break;
        }
        case API_Select::insert_and_evict: {
          avg_res[i].insert_and_evict_tm = sums[k++] / repeat_times;
          break;
        }
        default : {
          std::cout << "[Unsupport API!] \n";
        }
      }
    }
  }
  return avg_res;
}

void print_for_terminal(const std::vector<TestDescriptor> & tds,
                        const std::vector<API_Select>& target_apis) {
  // std::cout << "======================================\n";
  for (auto& td : tds) {
    ///TODO: format
    std::cout << "bucket_size   load_factor   hit_rate  dim   capacity(M)   hbm(GB)   hmem(GB)  hit_mode"
              << std::endl;
    std::string hit_mode = td.hit_mode == Hit_Mode::last_insert ? 
                          std::string{"last_insert"} : std::string{"random"};
    std::cout << setw(4) << setfill(' ') << td.bucket_size << rep(10)
              << fixed << setprecision(PRECISION) << td.load_factor << rep(9) 
              << fixed << setprecision(PRECISION) << td.hit_rate << rep(5) 
              << setw(4) << setfill(' ') << td.dim << rep(2) 
              << setw(4) << setfill(' ') << td.capacity/1024.0/1024 << rep(10) 
              << setw(4) << setfill(' ') << td.HBM4Values << rep(6) 
              << setw(4) << setfill(' ') << td.HMEM4Values << rep(5) 
              << hit_mode << std::endl;
    for (auto api : target_apis) {
      switch (api) {
        case API_Select::find: {
          std::cout << td.get_throughput(API_Select::find) * 100 
                    << "% \t";
          break;
        }
        case API_Select::insert_or_assign: {
          std::cout << td.get_throughput(API_Select::insert_or_assign) * 100 
                    << "% \t";
          break;
        }
        case API_Select::find_or_insert: {
          std::cout << td.get_throughput(API_Select::find_or_insert) * 100 
                    << "% \t";
          break;
        }
        case API_Select::assign: {
          std::cout << td.get_throughput(API_Select::assign) * 100 
                    << "% \t";
          break;
        }
        case API_Select::insert_and_evict: {
          std::cout << td.get_throughput(API_Select::insert_and_evict) * 100 
                    << "% \t";
          break;
        }
        default : {
          std::cout << "[Unsupport API!] \t";
        }
      }
    }
    std::cout << std::endl;
  }
}

void print_repeat_for_excel(const std::vector<std::vector<TestDescriptor>> & tds_vec,
                     const std::vector<API_Select>& target_apis) {
  std::cout << "======================================\n";
  auto repeat_times = tds_vec.size();
  if (repeat_times == 0) return;
  auto tests_kinds = tds_vec[0].size();
  
  for (int i = 0; i < tests_kinds; i++) {
    for (auto api : target_apis) {
      switch (api) {
        case API_Select::find: {
          for (int j = 0; j < repeat_times; j++) {
            std::cout << tds_vec[j][i].get_throughput(API_Select::find) * 100 
                      << "% / (" << tds_vec[j][i].find_tm << "ms) \t";
          }
          break;
        }
        case API_Select::insert_or_assign: {
          for (int j = 0; j < repeat_times; j++) {
            std::cout << tds_vec[j][i].get_throughput(API_Select::insert_or_assign) * 100 
                      << "% / (" << tds_vec[j][i].insert_or_assign_tm << "ms) \t";
          }
          break;
        }
        case API_Select::find_or_insert: {
          for (int j = 0; j < repeat_times; j++) {
            std::cout << tds_vec[j][i].get_throughput(API_Select::find_or_insert) * 100 
                      << "% / (" << tds_vec[j][i].find_or_insert_tm << "ms) \t";
          }
          break;
        }
        case API_Select::assign: {
          for (int j = 0; j < repeat_times; j++) {
            std::cout << tds_vec[j][i].get_throughput(API_Select::assign) * 100 
                      << "% / (" << tds_vec[j][i].assign_tm << "ms) \t";
          }
          break;
        }
        case API_Select::insert_and_evict: {
          for (int j = 0; j < repeat_times; j++) {
            std::cout << tds_vec[j][i].get_throughput(API_Select::insert_and_evict) * 100 
                      << "% / (" << tds_vec[j][i].insert_and_evict_tm << "ms) \t";
          }
          break;
        }
        default : {
          std::cout << "[Unsupport API!] ";
        }
      }
      std::cout << std::endl;
    }       
  }
}

int main(int argc, char* argv[]) {
  // benchmark::init_device(1, 0);
  try {
    {
#ifdef PROFILE
      // profiling and verify test
      std::vector<std::vector<TestDescriptor>> tds_vec;
      std::vector<API_Select> target_apis;
      // target_apis.push_back(API_Select::find);
      // target_apis.push_back(API_Select::insert_or_assign);
      // target_apis.push_back(API_Select::find_or_insert);
      // target_apis.push_back(API_Select::assign);
      target_apis.push_back(API_Select::find);
      for (int i = 0; i < kRepeat4Avg; i++) {
        std::cout << "============= loop " << i + 1 << " =============\n";
        std::vector<TestDescriptor> tds;
        // Add profiling detail
        // {
        //   TestDescriptor td;
        //   td.bucket_size = 8;
        //   td.load_factor = 0.1f;
        //   td.hit_rate = 1.0f;
        //   td.dim = 4;
        //   td.hit_mode = Hit_Mode::last_insert;
        //   tds.push_back(td);
        // }
        // {
        //   TestDescriptor td;
        //   td.bucket_size = 128;
        //   td.load_factor = 0.1f;
        //   td.hit_rate = 1.0f;
        //   td.hit_mode = Hit_Mode::last_insert;
        //   tds.push_back(td);
        // }
        // {
        //   TestDescriptor td;
        //   td.bucket_size = 128;
        //   td.load_factor = 0.1f;
        //   td.hit_rate = 0.6f;
        //   td.hit_mode = Hit_Mode::last_insert;
        //   tds.push_back(td);
        // }
        // {
        //   TestDescriptor td;
        //   td.bucket_size = 128;
        //   td.load_factor = 0.75f;
        //   td.hit_rate = 1.0f;
        //   td.hit_mode = Hit_Mode::last_insert;
        //   tds.push_back(td);     
        // }
        {
          TestDescriptor td;
          td.dim = 64;
          td.capacity = 15 * 1024 * 1024UL;
          td.key_num_per_op = 256 * 1024UL;
          td.bucket_size = 128;
          td.load_factor = 1.0;
          td.hit_rate = 1.0f;
          td.hit_mode = Hit_Mode::last_insert;
          tds.push_back(td);     
        }
        for (auto& td : tds) {
          for (auto api : target_apis) {
            testOneRound(api, td);
          }
        }
        print_for_terminal(tds, target_apis);
        tds_vec.push_back(tds);
      }
      auto&& avg_res = get_average_result(tds_vec, target_apis);
      std::cout << "# Average: \n";
      print_for_terminal(avg_res, target_apis);
      tds_vec.emplace_back(avg_res);
      std::cout << "# Excel: \n";
      print_repeat_for_excel(tds_vec, target_apis);
#else
      // contrast test
      std::vector<API_Select> target_apis;
      target_apis.push_back(API_Select::find);
      target_apis.push_back(API_Select::insert_or_assign);
      target_apis.push_back(API_Select::find_or_insert);
      target_apis.push_back(API_Select::assign);

      auto contrast_test_body = [&](const std::vector<TestDescriptor>& tds,
        bool sample_hit_rate = false) {
        std::vector<std::vector<TestDescriptor>> tds_vec;
        for (int i = 0; i < kRepeat4Avg; i++) {
          // avoid to modify original config
          auto tds_copy = tds;
          for (auto& td : tds_copy) {
            for (auto api : target_apis) {
              testOneRound(api, td, sample_hit_rate);
            }
          }
          tds_vec.push_back(tds_copy);
        }
        auto&& avg_res = get_average_result(tds_vec, target_apis);
        print_repeat_for_excel({avg_res}, target_apis);
      };
      TestDescriptor td;
      // common config
      td.capacity = 15 * 1024 * 1024UL;
      td.key_num_per_op = 256 * 1024UL;
      td.dim = 64;
      td.hit_rate = 0.6f;

      // std::cout << "================ Test hit_rate ====================\n";
      // auto old_hit_rate = td.hit_rate;
      // {
      //   std::cout << "=== Case bucket_size = 8, load_factor=0.1 ===\n";
      //   std::vector<TestDescriptor> tds;
      //   td.bucket_size = 8;
      //   td.load_factor = 0.1f;
      //   // repeat for different hit_rate
      //   for (int j = 0; j < 21; j++) {
      //     td.hit_rate = 1.0f * j / 20;
      //     tds.push_back(td);
      //   }
      //   contrast_test_body(tds, true);
      // }
      // {
      //   std::cout << "=== Case bucket_size = 128, load_factor=0.75 ===\n";
      //   std::vector<TestDescriptor> tds;
      //   td.bucket_size = 128;
      //   td.load_factor = 0.75f;
      //   for (int j = 0; j < 21; j++) {
      //     td.hit_rate = 1.0f * j / 20;
      //     tds.push_back(td);
      //   }
      //   contrast_test_body(tds, true);
      // }  
      // td.hit_rate = old_hit_rate;

      // std::cout << "================ Test dim ====================\n";
      // auto old_dim = td.dim;
      // {
      //   std::cout << "=== Case bucket_size = 8, load_factor=0.1 ===\n";
      //   std::vector<TestDescriptor> tds;
      //   td.bucket_size = 8;
      //   td.load_factor = 0.1f;
      //   // repeat for different dim
      //   for (int j = 2; j < 10; j++) {
      //     td.dim = (1 << j);
      //     tds.push_back(td);
      //   }
      //   contrast_test_body(tds);
      // }
      // {
      //   std::cout << "=== Case bucket_size = 128, load_factor=0.75 ===\n";
      //   std::vector<TestDescriptor> tds;
      //   td.bucket_size = 128;
      //   td.load_factor = 0.75f;
      //   for (int j = 2; j < 10; j++) {
      //     td.dim = (1 << j);
      //     tds.push_back(td);
      //   }
      //   contrast_test_body(tds);
      // }
      // td.dim = old_dim;

      std::cout << "================ Test bucket_size ====================\n";
      {
        std::cout << "=== case load_factor=0.1 ===\n";
        std::vector<TestDescriptor> tds;
        td.load_factor = 0.1f;
        for (int j = 1; j < 21; j++) {
          td.bucket_size = 8 * j;
          tds.push_back(td);
        }
        contrast_test_body(tds);
      }
      {
        std::cout << "=== case load_factor=0.75 ===\n";
        std::vector<TestDescriptor> tds;
        td.load_factor = 0.75f;
        for (int j = 1; j < 21; j++) {
          td.bucket_size = 8 * j;
          tds.push_back(td);
        }
        contrast_test_body(tds);
      }

      // to skip as cost too much time
      std::cout << "================ Test load_factor ====================\n";
      // {
      //   std::cout << "=== Case bucket_size = 8 ===\n";
      //   std::vector<TestDescriptor> tds;
      //   td.bucket_size = 8;
      //   // repeat for different load_factor
      //   for (int j = 1; j < 21; j++) {
      //     td.load_factor = static_cast<float>(1.0f * j / 20);
      //     tds.push_back(td);
      //   }
      //   contrast_test_body(tds);
      // }
      {
        std::cout << "=== Case bucket_size = 128 ===\n";
        std::vector<TestDescriptor> tds;
        td.bucket_size = 128;
        for (int j = 1; j < 21; j++) {
          td.load_factor = static_cast<float>(1.0f * j / 20);
          tds.push_back(td);
        }
        contrast_test_body(tds);
      }
#endif
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const nv::merlin::CudaException& e) {
    std::cerr << e.what() << endl;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
