/*
/usr/local/cuda-11.8/bin/nvcc CopyValueTest.cu -arch=sm_80; ./a.out
*/
#include <iostream>
#include <limits>
#include "../common.cuh"
#include "../../../include/merlin/core_kernels/kernel_utils.cuh"
using namespace std;
/*
A100 80G:　 2039  GB/s 
A100 40G:　 1555  GB/s 
3090    :   936.2 GB/s
*/
const float PeakBW = 2039.0f;
const int REPEAT = 20;
const uint64_t LEN = 128 * 1024 * 1024UL;


#define CHECK_CUDA(expr)                             \
{                                                    \
  auto status = (expr);                              \
  if (status != cudaSuccess) {                       \
    std::cout << "CUDA Error at "                    \
              << __FILE__ << ": "                    \
              << __LINE__ << "\n"                    \
              << cudaGetErrorName(status) << " "     \
              << cudaGetErrorString(status) << "\n"; \
  }                                                  \
}                                   

#define FETCH_FLOAT4(start) (reinterpret_cast<float4*>(&(start))[0])
// it seems that cuda only check if the base address whether out of bound
__global__ void memcpy(float* dst, float* src, size_t N) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) 
    FETCH_FLOAT4(dst[tid * 4]) = FETCH_FLOAT4(src[tid * 4]);
  // size_t stride = blockDim.x * gridDim.x;
  // for (size_t i = tid; i < N; i += stride) {
  //   FETCH_FLOAT4(dst[i * 4]) = FETCH_FLOAT4(src[i * 4]);
  // }
}

__device__ int dim_dev;

template <int32_t WARP_SIZE = 32>
__global__ void lookup_kernel_with_io_core_v5_2(float*  values, float**  values_addr, bool*  found, size_t n) {
  const int dim = 64;//dim_dev;
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  int rank = tid & 0x1f;
  int warpID = tid >> 5;
  const int WARP_NUM = (blockDim.x * gridDim.x) >> 5; 

  for (int v_idx = warpID; v_idx < n; v_idx += WARP_NUM) {
    if (found[v_idx]) {
      auto v_src = values_addr[v_idx];
      auto v_dst = values + v_idx * dim;

      for (auto j = rank; j < dim; j += WARP_SIZE) {
        v_dst[j] = v_src[j];
        // if (tid == 0) {
        //   printf("%f, %f \n", v_dst[j], v_src[j]);
        // }
      }      
    }
  }
}

template <int32_t TILE_SIZE = 16>
__global__ void lookup_kernel_with_io_core_v5_2_1(int* dim_dev_1, float*  values, float**  values_addr, bool*  found, size_t n) {

  const int dim = dim_dev_1[0];
  // const int dim = 64;//dim_dev;
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  int rank = tid % TILE_SIZE;
  int tileID = tid / TILE_SIZE;
  const int TILE_NUM = (blockDim.x * gridDim.x) / TILE_SIZE; 

  for (int v_idx = tileID; v_idx < n; v_idx += TILE_NUM) {
    if (found[v_idx]) {
      auto v_src = values_addr[v_idx];
      auto v_dst = values + v_idx * dim;
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
    }
  }
}

template <int32_t TILE_SIZE = 16, int32_t vector_per_block = 64>
__global__ void lookup_kernel_with_io_core_v5_2_2(float*  values, float**  values_addr, bool*  found, size_t n) {

  const int dim = 64;//dim_dev;
  int rank = threadIdx.x & (TILE_SIZE - 1);
  int tileID = threadIdx.x >> 4;

  // use int to avoid bank conflict
  __shared__ int found_Sm[vector_per_block];
  __shared__ float* src_addr_Sm[vector_per_block];
  int tx = threadIdx.x;
  if (tx < vector_per_block) {
    found_Sm[tx] = static_cast<int>(found[blockIdx.x * vector_per_block +  tx]);
    src_addr_Sm[tx] = values_addr[blockIdx.x * vector_per_block +  tx];
  }
  __syncthreads();


  const int TILE_NUM = (blockDim.x * gridDim.x) / TILE_SIZE; 

  for (int v_idx = tileID; v_idx < n; v_idx += TILE_NUM) {
    if (found_Sm[v_idx]) {
      auto v_src = src_addr_Sm[v_idx];
      float* v_dst = values + (blockIdx.x * vector_per_block + v_idx)  * dim;
      FETCH_FLOAT4(v_dst[rank * 4]) = FETCH_FLOAT4(v_src[rank * 4]);
    }
  }
}


__global__ void set_addr(int* dim_dev_1, float** addrs, float* value, int n, int dim) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    addrs[tid] = value + tid * dim;
  }
  if (tid == 0) {
    dim_dev = 64;
    dim_dev_1[0] = 64;
  }
}
__global__ void mem_check(float* value, size_t len, float val) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < len) {
    float distance = value[tid] - val;
    if (distance > 1e-3 || distance < -1 * 1e-3) {
        printf("%f \n",value[tid]);
    }
  }
}
__global__ void mem_set(float* value, size_t len, float val) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < len) {
    value[tid] = val;
  }
}

__global__ void copy_large_memSpace(float* src, float* dst, int N, int stride) {

  constexpr 

}
void test_copy_vector() {
  constexpr size_t dim = 64;
  const size_t n =  LEN / dim;

  if (dim > 128) {
    std::cout << "Dim of vector is too large\n";
    exit(-1);
  }

  constexpr int threads_per_vector = dim / 4;
  int32_t block_size = 1024;
  int32_t vectors_per_block = block_size / (dim / 4);
  int needed_block_num = n / vectors_per_block;

  int grid_size = 4096 > needed_block_num ? needed_block_num : 4096;

  float* values_dst;
  float* values_src;
  float** values_addr;
  bool* found;
  float** addr_h;
  int* dim_dev_1;
  CHECK_CUDA(cudaMalloc(&values_src, LEN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&values_dst, LEN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&values_addr, n * sizeof(float*)));
  CHECK_CUDA(cudaMalloc(&found, n * sizeof(bool)));
  CHECK_CUDA(cudaMalloc(&dim_dev_1, sizeof(int)));
  addr_h = (float**)malloc(n * sizeof(float*));

  mem_set<<<LEN / block_size, block_size>>>(values_src, LEN, 1.0f);
  mem_set<<<LEN / block_size, block_size>>>(values_dst, LEN, 2.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  mem_check<<<LEN / block_size, block_size>>>(values_src, LEN, 1.0f);
  mem_check<<<LEN / block_size, block_size>>>(values_dst, LEN, 2.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  set_addr<<<n / block_size, block_size>>>(dim_dev_1, values_addr, values_src, n, dim);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(addr_h, values_addr, n * sizeof(float*), cudaMemcpyDeviceToHost));

  // std::cout << "Start address: " << values_src << "\n";
  // std::cout << "Vector base addresses: ";
  // for (int i = 0; i < 5; i++) {
  //   auto expected_addr = values_src + i * dim;
  //   std::cout  << addr_h[i] << " / " << expected_addr << "\n";
  // }
  // std::cout << "Last address: " << addr_h[n - 1] << " / " << values_src + (n - 1) * dim << "\n";

  CHECK_CUDA(cudaMemset(found, 1, n * sizeof(bool)));
  CHECK_CUDA(cudaMemcpyToSymbol(dim_dev, &dim, sizeof(int)));

  float time;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  // lookup_kernel_with_io_core_v5_2<<<n/32, block_size>>>(values_dst, values_addr, found, n);

  lookup_kernel_with_io_core_v5_2_1<threads_per_vector>
            <<<grid_size, block_size>>>(dim_dev_1, values_dst, values_addr, found, n);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaDeviceSynchronize());

  mem_check<<<LEN / block_size, block_size>>>(values_dst, LEN, 1.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  float sol = 2.0 * LEN * sizeof(float)/(1024 * 1024 * 1024.0) / (time/1000);
  std::cout << "SOL = " << sol
            << " GB/s \t" << sol / GPUMemBandwidth * 100 << "%" << std::endl;

  CHECK_CUDA(cudaFree(values_src));  
  CHECK_CUDA(cudaFree(values_dst));
  CHECK_CUDA(cudaFree(values_addr));
  CHECK_CUDA(cudaFree(found));
}

void test_memcpy_kernel() {
  const uint64_t block_size = 512;//1024;
  const uint64_t grid_size = 4096; // tail effect

  if (LEN < block_size * grid_size * 4) {
    std::cout << "Var <LEN> is small\n";
  }
  if (LEN % 4 != 0) {
    std::cout << "Address will be out of bounds\n";
  }
  float *dev_src, *dev_dst;
  float time;
  CHECK_CUDA(cudaMalloc(&dev_src, LEN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_dst, LEN * sizeof(float)));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  memcpy<<<LEN / 4 / block_size/*grid_size*/, block_size>>>(dev_dst, dev_src, (LEN + 3) / 4);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  float sol = 2.0 * LEN * sizeof(float)/(1024 * 1024 * 1024) / (time/1000);
  std::cout << "SOL = " << sol << " GB/s \t" 
            << sol / GPUMemBandwidth * 100 << "%" << std::endl;
  CHECK_CUDA(cudaFree(dev_src));
  CHECK_CUDA(cudaFree(dev_dst));
}
void test_memcpy_runtime_api() {
  float *dev_src, *dev_dst;
  float time;
  CHECK_CUDA(cudaMalloc(&dev_src, LEN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_dst, LEN * sizeof(float)));
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  CHECK_CUDA(cudaMemcpyAsync(dev_dst, dev_src, LEN * sizeof(float), cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  float sol = 2.0 * LEN * sizeof(float)/(1024 * 1024 * 1024) / (time/1000);
  std::cout << "SOL = " << sol << " GB/s \t" 
            << sol / GPUMemBandwidth * 100 << "%" << std::endl;
  CHECK_CUDA(cudaFree(dev_src));
  CHECK_CUDA(cudaFree(dev_dst));  
}
int main() {
  cudaSetDevice(1);
  cudaFree(0);

  std::cout << "=========== Runtime API ===================\n";
  for (int i = 0; i < REPEAT; i++)
    test_memcpy_runtime_api();

  std::cout << "=========== Memcpy kernel =================\n";
  for (int i = 0; i < REPEAT; i++)
    test_memcpy_kernel();

  std::cout << "=========== Copy Vector ===================\n";
  for (int i = 0; i < REPEAT; i++)
    test_copy_vector();
}
/* A100 80 GB
=========== Runtime API ===================
SOL = 1089.91 GB/s      53.4533%
SOL = 1284.95 GB/s      63.0187%
SOL = 1284.95 GB/s      63.0187%
SOL = 1284.95 GB/s      63.0187%
SOL = 1298.62 GB/s      63.6891%
SOL = 1312.58 GB/s      64.3739%
SOL = 1271.57 GB/s      62.3622%
SOL = 1312.58 GB/s      64.3739%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
SOL = 1312.58 GB/s      64.3739%
SOL = 1312.58 GB/s      64.3739%
SOL = 1284.95 GB/s      63.0187%
SOL = 1312.58 GB/s      64.3739%
SOL = 1298.62 GB/s      63.6891%
SOL = 1298.62 GB/s      63.6891%
=========== Memcpy kernel =================
SOL = 1371.58 GB/s      67.2671%
SOL = 1403.11 GB/s      68.8135%
SOL = 1403.11 GB/s      68.8135%
SOL = 1403.11 GB/s      68.8135%
SOL = 1436.12 GB/s      70.4326%
SOL = 1436.12 GB/s      70.4326%
SOL = 1419.42 GB/s      69.6136%
SOL = 1436.12 GB/s      70.4326%
SOL = 1436.12 GB/s      70.4326%
SOL = 1403.11 GB/s      68.8135%
SOL = 1419.42 GB/s      69.6136%
SOL = 1419.42 GB/s      69.6136%
SOL = 1453.22 GB/s      71.2711%
SOL = 1436.12 GB/s      70.4326%
SOL = 1419.42 GB/s      69.6136%
SOL = 1436.12 GB/s      70.4326%
SOL = 1419.42 GB/s      69.6136%
SOL = 1419.42 GB/s      69.6136%
SOL = 1403.11 GB/s      68.8135%
SOL = 1436.12 GB/s      70.4326%
=========== Copy Vector ===================
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1356.34 GB/s      66.5197%
SOL = 1341.43 GB/s      65.7887%
SOL = 1356.34 GB/s      66.5197%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1356.34 GB/s      66.5197%
SOL = 1356.34 GB/s      66.5197%
SOL = 1356.34 GB/s      66.5197%
SOL = 1326.85 GB/s      65.0736%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1341.43 GB/s      65.7887%
SOL = 1507.04 GB/s      73.9108%
=========== Copy Vector(use shared) ===================
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1371.58 GB/s      67.2671%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1371.58 GB/s      67.2671%
SOL = 1371.58 GB/s      67.2671%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1387.16 GB/s      68.0315%
SOL = 1565 GB/s         76.7535%
root@3b8a3785ef52:/Workspace/HierarchicalKV/include/merlin/lookup_kernels# 
*/

/*3090
=========== Runtime API ===================
SOL = 65.0413 GB/s      6.94737%
SOL = 767.738 GB/s      82.0057%
SOL = 777.518 GB/s      83.0504%
SOL = 772.597 GB/s      82.5248%
SOL = 772.597 GB/s      82.5248%
SOL = 782.502 GB/s      83.5828%
SOL = 777.518 GB/s      83.0504%
SOL = 782.502 GB/s      83.5828%
SOL = 782.973 GB/s      83.633%
SOL = 782.502 GB/s      83.5828%
SOL = 777.518 GB/s      83.0504%
SOL = 787.55 GB/s       84.122%
SOL = 777.518 GB/s      83.0504%
SOL = 772.597 GB/s      82.5248%
SOL = 779.069 GB/s      83.216%
SOL = 782.502 GB/s      83.5828%
SOL = 777.518 GB/s      83.0504%
SOL = 777.828 GB/s      83.0835%
SOL = 778.448 GB/s      83.1497%
SOL = 782.502 GB/s      83.5828%
=========== Memcpy kernel =================
SOL = 748.898 GB/s      79.9933%
SOL = 782.502 GB/s      83.5828%
SOL = 787.55 GB/s       84.122%
SOL = 792.664 GB/s      84.6683%
SOL = 788.823 GB/s      84.2579%
SOL = 789.141 GB/s      84.292%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 787.55 GB/s       84.122%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 787.55 GB/s       84.122%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 792.664 GB/s      84.6683%
SOL = 782.502 GB/s      83.5828%
SOL = 787.55 GB/s       84.122%
SOL = 787.55 GB/s       84.122%
=========== Copy Vector ===================
SOL = 768.191 GB/s      82.0541%
SOL = 767.738 GB/s      82.0057%
SOL = 762.939 GB/s      81.4932%
SOL = 744.331 GB/s      79.5056%
SOL = 772.75 GB/s       82.5411%
SOL = 772.597 GB/s      82.5248%
SOL = 769.25 GB/s       82.1672%
SOL = 772.597 GB/s      82.5248%
SOL = 772.597 GB/s      82.5248%
SOL = 772.597 GB/s      82.5248%
SOL = 762.939 GB/s      81.4932%
SOL = 768.796 GB/s      82.1187%
SOL = 772.597 GB/s      82.5248%
SOL = 768.191 GB/s      82.0541%
SOL = 777.518 GB/s      83.0504%
SOL = 771.072 GB/s      82.3619%
SOL = 768.493 GB/s      82.0864%
SOL = 777.518 GB/s      83.0504%
SOL = 773.056 GB/s      82.5738%
SOL = 767.738 GB/s      82.0057%
*/