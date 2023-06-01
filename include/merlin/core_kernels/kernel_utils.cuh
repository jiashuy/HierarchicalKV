/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <mutex>
#include <thread>
#include <vector>
#include "../types.cuh"
#include "../utils.cuh"

// if i % 2 == 0, select buffer 0, else buffer 1
#define SAME_BUF(i) (((i)&0x01) ^ 0)
// if i % 2 == 0, select buffer 1, else buffer 0
#define DIFF_BUF(i) (((i)&0x01) ^ 1)

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace nv {
namespace merlin {

template <typename S, typename K, int BUCKET_SIZE = 128>
struct CopyScoreEmpty {
  __forceinline__ __device__ static S* get_base_ptr(K** keys_ptr, int offset) {
    return nullptr;
  }
  __forceinline__ __device__ static void ldg_sts(S* dst, S* src) {}
  __forceinline__ __device__ static S lgs(S* src) { return 0; }
  __forceinline__ __device__ static void stg(S* dst, S score_) {}
};

template <typename S, typename K, int BUCKET_SIZE = 128>
struct CopyScoreByPassCache {
  __forceinline__ __device__ static S* get_base_ptr(K** keys_ptr, int offset) {
    return reinterpret_cast<S*>(keys_ptr[offset] + BUCKET_SIZE);
  }

  __forceinline__ __device__ static void ldg_sts(S* dst, S* src) {
    __pipeline_memcpy_async(dst, src, sizeof(S));
  }

  __forceinline__ __device__ static S lgs(S* src) { return src[0]; }

  __forceinline__ __device__ static void stg(S* dst, S score_) {
    __stwt(dst, score_);
  }
};

template <typename V = float, typename CopyUnit = float4, int GROUP_SIZE = 16,
          int UnitSize = sizeof(CopyUnit) / sizeof(V)>
struct CopyValueOneGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, V* dst, V* src,
                                                 int dim) {
    int offset = rank * UnitSize;
    if (offset < dim)
      __pipeline_memcpy_async(dst + offset, src + offset, sizeof(CopyUnit));
  }

  __forceinline__ __device__ static void lds_stg(int rank, V* dst, V* src,
                                                 int dim) {
    int offset = rank * UnitSize;
    if (offset < dim) {
      CopyUnit value_ = reinterpret_cast<CopyUnit*>(src + offset)[0];
      __stwt(reinterpret_cast<CopyUnit*>(dst + offset), value_);
    }
  }
};

template <typename V = float, typename CopyUnit = float4, int GROUP_SIZE = 16,
          int UnitSize = sizeof(CopyUnit) / sizeof(V)>
struct CopyValueTwoGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, V* dst, V* src,
                                                 int dim) {
    int offset = rank * UnitSize;
    __pipeline_memcpy_async(dst + offset, src + offset, sizeof(CopyUnit));
    offset += GROUP_SIZE * UnitSize;
    if (offset < dim)
      __pipeline_memcpy_async(dst + offset, src + offset, sizeof(CopyUnit));
  }

  __forceinline__ __device__ static void lds_stg(int rank, V* dst, V* src,
                                                 int dim) {
    int offset = rank * UnitSize;
    CopyUnit value_ = reinterpret_cast<CopyUnit*>(src + offset)[0];
    __stwt(reinterpret_cast<CopyUnit*>(dst + offset), value_);
    offset += GROUP_SIZE * UnitSize;
    if (offset < dim) {
      CopyUnit value_ = reinterpret_cast<CopyUnit*>(src + offset)[0];
      __stwt(reinterpret_cast<CopyUnit*>(dst + offset), value_);
    }
  }
};

template <typename V = float, typename CopyUnit = float4, int GROUP_SIZE = 16,
          int UnitSize = sizeof(CopyUnit) / sizeof(V),
          int STRIDE = GROUP_SIZE* UnitSize>
struct CopyValueMultipleGroup {
  __forceinline__ __device__ static void ldg_sts(int rank, V* dst, V* src,
                                                 int dim) {
    for (int offset = rank * UnitSize; offset < dim; offset += STRIDE) {
      __pipeline_memcpy_async(dst + offset, src + offset, sizeof(CopyUnit));
    }
  }

  __forceinline__ __device__ static void lds_stg(int rank, V* dst, V* src,
                                                 int dim) {
    for (int offset = rank * UnitSize; offset < dim; offset += STRIDE) {
      CopyUnit value_ = reinterpret_cast<CopyUnit*>(src + offset)[0];
      __stwt(reinterpret_cast<CopyUnit*>(dst + offset), value_);
    }
  }
};

}  // namespace merlin
}  // namespace nv
