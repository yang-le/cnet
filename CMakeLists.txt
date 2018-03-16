cmake_minimum_required(VERSION 3.9)

project(cnet)

include_directories(core layer opencl cuda gflags)

file(GLOB core "core/*.c")
file(GLOB layer "layer/*.c")

option(USE_OPENMP "use openmp to speed up gemm" OFF)
option(USE_BLAS "use blas to speed up gemm" OFF)
option(USE_CLBLAS "use clblas to speed up gemm" OFF)
option(USE_CLBLAST "use clblast to speed up gemm" OFF)
option(USE_CUDA "use cuda to speed up gemm" OFF)

option(USE_OPENCV "use opencv as gui" OFF)

option(USE_GFLAGS "use gflags" OFF)

if (UNIX)
set(libs m)
endif()

if(USE_OPENMP)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(libs ${libs} ${OpenMP_C_LIBRARIES})
endif()

if(USE_BLAS)
find_package(BLAS REQUIRED)
add_definitions(-DUSE_BLAS)
set(libs ${libs} ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()

if(USE_CLBLAST)
find_package(CLBlast REQUIRED)
add_definitions(-DUSE_OPENCL)
add_definitions(-DUSE_CLBLAST)
file(GLOB opencl "opencl/*.c")
set(libs ${libs} clblast)
endif()

if(USE_CLBLAS)
find_package(clBLAS REQUIRED)
add_definitions(-DUSE_OPENCL)
add_definitions(-DUSE_CLBLAS)
file(GLOB opencl "opencl/*.c")
set(libs ${libs} clBLAS)
endif()

if(USE_CUDA)
find_package(CUDA REQUIRED)
add_definitions(-DUSE_CUDA)
file(GLOB cuda "cuda/*.c")
include_directories(${CUDA_INCLUDE_DIRS})
set(libs ${libs} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

if(USE_OPENCV)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)
set(libs ${libs} ${OpenCV_LIBS})
endif()

if(USE_GFLAGS)
find_package(GFlags REQUIRED)
add_definitions(-DUSE_GFLAGS)
file(GLOB gflags "gflags/*.cc")
set(libs ${libs} gflags)
endif()

add_library(cnet ${core} ${layer} ${opencl} ${cuda} ${gflags})
target_link_libraries(cnet ${libs})

add_executable(linear_example example/linear_example.c)
target_link_libraries(linear_example cnet)

add_executable(mnist_example example/mnist_example.c example/mnist.c)
target_link_libraries(mnist_example cnet)

add_executable(xor_example example/xor_example.c)
target_link_libraries(xor_example cnet)

add_executable(vgg_example example/vgg_example.c)
target_link_libraries(vgg_example cnet)

add_executable(googlenet_example example/googlenet_example.c example/mnist.c)
target_link_libraries(googlenet_example cnet)
