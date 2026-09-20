cmake_minimum_required(VERSION 3.22)
project(CTransformer)

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(ARCH_IS_ARM TRUE)
else()
  set(ARCH_IS_ARM FALSE)
endif()

option(NEON_OPT "enable neon optimizations" ${ARCH_IS_ARM})
if (NEON_OPT)
  add_compile_definitions(NEON_OPT=true)
endif()

option(RKNN_BACKEND "use the RKNN NPU backend for Matmul" ${ARCH_IS_ARM})
if (RKNN_BACKEND)
  add_compile_definitions(HAVE_RKNN=true)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fPIC -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fPIC -march=native")

enable_testing()

add_subdirectory(lib)
add_subdirectory(examples/whisper)
