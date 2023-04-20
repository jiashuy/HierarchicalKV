
#include "v1.cuh"
#include "v2.cuh"
#include "v3.cuh"
#include "v4.cuh"


template <class K, class V, class M, uint32_t iteration = 4, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel_with_io_beta(const Table<K, V, M>* table,
                                      const K* __restrict keys,
                                      V* __restrict values, M* __restrict metas,
                                      bool* __restrict found, size_t N) {
//-----------------------------------------------------------------------------
/* 
  no atmoic
*/

  lookup_kernel_with_io_core_v1<K, V, M, TILE_SIZE>(table, keys, values, metas, found, N);

//-----------------------------------------------------------------------------
/*
  no atomic
  no function call
  use __ballot_sync instead of ballot : no help
  prefetch meta to register
  align to cacheline size to avoid uncoalesed memory access : don't fit with all case
  use the specail key instead of lock the bucket
*/

  // lookup_kernel_with_io_core_v2<K, V, M, TILE_SIZE>(table, keys, values, metas, found, N);

//-----------------------------------------------------------------------------
/*
  no atomic
  no function call
  use __ballot_sync instead of ballot : no help
  prefetch meta to register
  align to cacheline size to avoid uncoalesed memory access : don't fit with all case
  use the specail key instead of lock the bucket
  
  unroll the code for [iteration] times
*/
  // lookup_kernel_with_io_core_v3<K, V, M, iteration>(table, keys, values, metas, found, N);

//-----------------------------------------------------------------------------

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

  // lookup_kernel_with_io_core_v4<K, V, M, iteration>(table, keys, values, metas, found, N);
}

template <typename K, typename V, typename M>
void lookup_kernel_beta(cudaStream_t& stream, const size_t& n,
                        const Table<K, V, M>* __restrict table,
                        const K* __restrict keys, V* __restrict values,
                        M* __restrict metas, bool* __restrict found) {
      
  const int iteration = 4;
  unsigned int block_size = 128;



  // if not unroll
  const unsigned int tile_size = 32;
  const size_t N = n * tile_size;
  const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

  // if unroll
  // const unsigned int tile_size = 32;
  // const size_t grid_size = n / block_size;
  // const size_t N = n * tile_size / iteration;   



  lookup_kernel_with_io_beta<K, V, M, iteration, tile_size>
    <<<grid_size, block_size, 0, stream>>>(table, keys, values, metas,
                                            found, N);
}