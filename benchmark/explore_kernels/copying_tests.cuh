#pragma once
#include "common.cuh"
#include "merlin/core_kernels/kernel_utils.cuh"
using namespace nv::merlin;

// A GROUP copy a value
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_copying_origin(
    const Table<K, V, S>* __restrict table, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {
  
  int dim = table->dim;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int rank = tid & (GROUP_SIZE - 1);
  int key_idx = tid >> MASK_WIDTH;
  if (key_idx >= n) return;

  auto v_src = values_addr[key_idx];
  auto v_dst = values + key_idx * dim;
  if (founds[key_idx]) {
    FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A GROUP copy a value
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int rank = tid & (GROUP_SIZE - 1);
  int key_idx = tid >> MASK_WIDTH;
  if (key_idx >= n) return;

  auto v_src = values_addr[key_idx];
  auto v_dst = values + key_idx * dim;
  if (founds[key_idx]) {
    FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A GROUP copy GROUP_SIZE values
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2_test1(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {
  constexpr int BLOCK_SIZE = 128;
  __shared__ V* sm_values_addr[BLOCK_SIZE];
  __shared__ int sm_founds[BLOCK_SIZE];

  int groupID = threadIdx.x >> MASK_WIDTH;
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;

  int loop_num = n - key_idx_base < GROUP_SIZE ? n - key_idx_base : GROUP_SIZE;
  if (loop_num <= 0) return; 
  int rank = threadIdx.x & (GROUP_SIZE - 1);
  if (rank < loop_num) {
    sm_values_addr[groupID * GROUP_SIZE + rank] = values_addr[key_idx_base + rank];
    sm_founds[groupID * GROUP_SIZE + rank] = static_cast<int>(founds[key_idx_base + rank]);
  }
  for (int i = 0; i < loop_num; i++) {
    auto v_src = sm_values_addr[groupID * GROUP_SIZE + i];
    auto v_dst = values + (key_idx_base + i) * dim;
    if (sm_founds[groupID * GROUP_SIZE + i])
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A GROUP copy GROUP_SIZE values: 
// prefetch using async copy, write back using cache hint
template <class K, class V, class S, 
          int32_t GROUP_SIZE = 32, int32_t MASK_WIDTH = 5>
__global__ void lookup_kernel_with_io_v2_kernel2_test2(
    const int dim, V* __restrict values, V** __restrict values_addr, 
    bool* __restrict founds, size_t n) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;
  __shared__ V* sm_values_addr[BLOCK_SIZE];
  __shared__ int sm_founds[BLOCK_SIZE];
  __shared__ V sm_values[2][GROUP_NUM][GROUP_SIZE * 4];

  int groupID = threadIdx.x >> MASK_WIDTH;
  int key_idx_base = (blockIdx.x * blockDim.x) + groupID * GROUP_SIZE;

  int loop_num = n - key_idx_base < GROUP_SIZE ? n - key_idx_base : GROUP_SIZE;
  if (loop_num <= 0) return; 
  int rank = threadIdx.x & (GROUP_SIZE - 1);
  if (rank < loop_num) {
    int key_idx_block = groupID * GROUP_SIZE + rank;
    int key_idx_grid = key_idx_base + rank;
    __pipeline_memcpy_async(sm_values_addr + key_idx_block, values_addr + key_idx_grid,
                            sizeof(V*));
    __pipeline_commit();
    sm_founds[key_idx_block] = static_cast<int>(founds[key_idx_grid]);
  }
  __pipeline_wait_prior(0);
  V* v_src = sm_values_addr[groupID * GROUP_SIZE];
  V* v_dst = sm_values[0][groupID];
  __pipeline_memcpy_async(v_dst, 
                          v_src,
                          sizeof(float4));
  __pipeline_commit();
  for (int i = 0; i < loop_num; i++) {
    if ((i + 1) < loop_num) {
      V* v_dst = sm_values[(i & 0x01) ^ 1][groupID];
      V* v_src = sm_values_addr[groupID * GROUP_SIZE + i + 1];
      __pipeline_memcpy_async(v_dst + rank * 4, 
                              v_src + rank * 4,
                              sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    V* v_src = sm_values[(i & 0x01) ^ 0][groupID];
    V* v_dst = values + (key_idx_base + i) * dim;
    if (sm_founds[groupID * GROUP_SIZE + i]) {
      float4 value_ = FETCH_FLOAT4(v_src[rank * 4]);
      __stwt(reinterpret_cast<float4*>(v_dst + rank * 4), value_);
    }
  }
}

// Test Runtime API : cudaMemcpyAsync, contrast test
template<typename V, int REPEAT>
float test_cudaMemcpyAsync(const size_t LEN) {
  V *dev_src, *dev_dst;
  float time;
  CUDA_CHECK(cudaMalloc(&dev_src, LEN * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&dev_dst, LEN * sizeof(V)));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  #pragma unroll
  for (int i = 0; i < REPEAT; i++) {
    CUDA_CHECK(cudaMemcpyAsync(dev_dst, dev_src, LEN * sizeof(V), cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dev_src));
  CUDA_CHECK(cudaFree(dev_dst));  
  return static_cast<float>(time / REPEAT);
}

// A thread copy a float4
__global__ void memcpy_kernel1(float* dst, float* src, size_t N) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = tid * 4;
  if (offset < N) 
    FETCH_FLOAT4(dst[offset]) = FETCH_FLOAT4(src[offset]);
}

// A thread copy a float4, bypass cache
__global__ void memcpy_kernel2(float* dst, float* src, size_t N) {
  constexpr int BLOCK_SIZE = 128;
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sm_buffer[BLOCK_SIZE * 4];
  int offset = tid * 4;
  if (offset < N) {
    __pipeline_memcpy_async(sm_buffer + threadIdx.x * 4, 
                            src + offset,
                            sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    float4 value_ = FETCH_FLOAT4(sm_buffer[threadIdx.x * 4]);
    __stwt(reinterpret_cast<float4*>(dst + offset), value_);
  }
}

// A group copy for `TIMES` times, with every thread copy a float4 per operation
template<int GROUP_SIZE = 32, int TIMES = GROUP_SIZE>
__global__ void memcpy_kernel3(float* dst, float* src, size_t N) {
  constexpr int LOOP_STRIDE = GROUP_SIZE * 4;
  constexpr int GROUP_STRIDE = LOOP_STRIDE * TIMES;

  static_assert(GROUP_STRIDE == GROUP_SIZE * 128);

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int groupID = tid / GROUP_SIZE;
  int groupBase = groupID * GROUP_STRIDE;
  if (groupBase >= N) return;
  int rank = tid % GROUP_SIZE;

  for (int i = 0; i < TIMES; i++) {
    float* v_dst = dst + (groupBase + i * LOOP_STRIDE);
    float* v_src = src + (groupBase + i * LOOP_STRIDE);
    FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
  }
}

// A group copy for `TIMES` times, with every thread copy a float4 per operation
// Aync Copy
template<int GROUP_SIZE = 32, int TIMES = GROUP_SIZE>
__global__ void memcpy_kernel4(float* dst, float* src, size_t N) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int LOOP_STRIDE = GROUP_SIZE * 4;
  constexpr int GROUP_STRIDE = LOOP_STRIDE * TIMES;
  constexpr int GROUP_NUM = BLOCK_SIZE / GROUP_SIZE;

  static_assert(GROUP_STRIDE == GROUP_SIZE * 128);

  __shared__ float sm_buffer[2][GROUP_NUM][GROUP_SIZE * 4];

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int groupID = tid / GROUP_SIZE;
  int groupBase = groupID * GROUP_STRIDE;
  groupID = threadIdx.x / GROUP_SIZE;
  if (groupBase >= N) return;
  int rank = tid % GROUP_SIZE;

  float* v_dst = dst + groupBase;
  float* v_src = src + groupBase;
  __pipeline_memcpy_async(sm_buffer[0][groupID] + rank * 4, 
                          v_src + rank * 4,
                          sizeof(float4));
  __pipeline_commit();

  for (int i = 0; i < TIMES; i++) {
    if ((i + 1) < TIMES) {
      __pipeline_memcpy_async(sm_buffer[(i & 0x01) ^ 1][groupID] + rank * 4, 
                        v_src + (i+1) * LOOP_STRIDE  + rank * 4,
                        sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    float4 value_ = FETCH_FLOAT4(sm_buffer[(i & 0x01) ^ 0][groupID][rank * 4]);
    int offset = i * LOOP_STRIDE + rank * 4;
    __stwt(reinterpret_cast<float4*>(v_dst + offset), value_);
  }
}

__global__ void memset_kernel(float* arr, float val) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  arr[tid] = val;
}

__global__ void memcheck_kernel(float* arr, float val) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (arr[tid] != val) {
    printf("unequal ");
  }
}

// Test memcpy kernel : contrast test
template<typename V, int REPEAT>
float test_memcpy_kernel(const size_t LEN) {
  const int dim = 128;
  const int nums = LEN / dim;
  constexpr int BLOCK_SIZE = 128;
  V *dev_src, *dev_dst;
  float time;
  CUDA_CHECK(cudaMalloc(&dev_src, LEN * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&dev_dst, LEN * sizeof(V)));
  memset_kernel<<<LEN / BLOCK_SIZE, BLOCK_SIZE>>>(dev_src, 1.23f);
  memset_kernel<<<LEN / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, 0.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  #pragma unroll
  for (int i = 0; i < REPEAT; i++) {
    //////////////////////////////////////////////////////////////////////////////////////
    // memcpy_kernel1<<< LEN / 4 / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //////////////////////////////////////////////////////////////////////////////////////
    // memcpy_kernel2<<< LEN / 4 / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //////////////////////////////////////////////////////////////////////////////////////
    //---------------------
    // memcpy_kernel3<4, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    // memcpy_kernel3<8, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    // memcpy_kernel3<16, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    memcpy_kernel3<32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //////////////////////////////////////////////////////////////////////////////////////
    //---------------------
    // memcpy_kernel4<4, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    // memcpy_kernel4<8, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    // memcpy_kernel4<16, 32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
    //---------------------
    // memcpy_kernel4<32><<< nums / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, dev_src, LEN);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  memcheck_kernel<<<LEN / BLOCK_SIZE, BLOCK_SIZE>>>(dev_dst, 1.23f);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(dev_src));
  CUDA_CHECK(cudaFree(dev_dst));  
  return static_cast<float>(time / REPEAT);
}