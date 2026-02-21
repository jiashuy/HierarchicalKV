cmake_minimum_required(VERSION 3.18)
project(e8_baselines LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GPU architecture (default: Ampere sm_86 for RTX A6000)
set(GPU_ARCH "86" CACHE STRING "CUDA architecture")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA compile flags needed by dependencies
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

include(FetchContent)

# Disable tests/examples/benchmarks from all dependencies
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BGHT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CUCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# ─── WarpCore ───
FetchContent_Declare(warpcore
  GIT_REPOSITORY https://github.com/sleeepyjack/warpcore.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)

# ─── BGHT (header-only, just download) ───
FetchContent_Declare(bght
  GIT_REPOSITORY https://github.com/owensgroup/BGHT.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE
)

# ─── cuCollections ───
FetchContent_Declare(cuco
  GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
  GIT_TAG        dev
  GIT_SHALLOW    TRUE
)

# ─── WarpCore benchmark ───
message(STATUS "Fetching WarpCore ...")
FetchContent_MakeAvailable(warpcore)
add_executable(warpcore_bench warpcore_bench.cu)
target_include_directories(warpcore_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(warpcore_bench PRIVATE warpcore)
set_target_properties(warpcore_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ─── BGHT benchmark (header-only: populate manually, skip its CMake) ───
message(STATUS "Fetching BGHT ...")
FetchContent_GetProperties(bght)
if(NOT bght_POPULATED)
  FetchContent_Populate(bght)
endif()
add_executable(bght_bench bght_bench.cu)
target_include_directories(bght_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${bght_SOURCE_DIR}/include
)
set_target_properties(bght_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ─── cuCollections benchmark ───
message(STATUS "Fetching cuCollections ...")
FetchContent_MakeAvailable(cuco)
add_executable(cuco_bench cuco_bench.cu)
target_include_directories(cuco_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuco_bench PRIVATE cuco::cuco)
set_target_properties(cuco_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
