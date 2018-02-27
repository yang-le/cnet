cmake_minimum_required(VERSION 3.9)

project(cnet)

include_directories(core layer opencl)

file(GLOB core "core/*.c")
file(GLOB layer "layer/*.c")

option(USE_OPENMP "use openmp to speed up gemm")
option(USE_BLAS "use blas to speed up gemm")

if(USE_OPENMP)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
endif()

if(USE_BLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
add_definitions(-DUSE_BLAS)
endif()
endif()

if(USE_OPENCL)
find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
add_definitions(-DUSE_OPENCL)
add_subdirectory(third_party/clBLAS/src)
file(GLOB opencl "opencl/*.c")
endif()
endif()

add_library(cnet ${core} ${layer} ${opencl})
if(BLAS_FOUND)
target_link_libraries(cnet ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()
if (OpenCL_FOUND)
target_link_libraries(cnet clBLAS)
endif()

add_executable(linear_example example/linear_example.c)
target_link_libraries(linear_example cnet m)

add_executable(mnist_example example/mnist_example.c example/mnist.c)
target_link_libraries(mnist_example cnet m)

add_executable(xor_example example/xor_example.c)
target_link_libraries(xor_example cnet m)
