cmake_minimum_required(VERSION 3.9)

project(cnet)

include_directories(core layer opencl cuda)

file(GLOB core "core/*.c")
file(GLOB layer "layer/*.c")

option(USE_OPENMP "use openmp to speed up gemm" OFF)
option(USE_BLAS "use blas to speed up gemm" OFF)
option(USE_CLBLAS "use clblas to speed up gemm" OFF)
option(USE_CLBLAST "use clblast to speed up gemm" OFF)
option(USE_CUDA "use cuda to speed up gemm" OFF)

option(USE_OPENCV "use opencv as gui" OFF)

if (UNIX)
set(libs m)
endif()

if(USE_OPENMP)
find_package(OpenMP REQUIRED)
if(OPENMP_C_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(libs ${libs} ${OpenMP_C_LIBRARIES})
endif()
endif()

if(USE_BLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
add_definitions(-DUSE_BLAS)
set(libs ${libs} ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()
endif()

if(USE_CLBLAST)
find_package(CLBlast REQUIRED)
if (CLBlast_FOUND)
add_definitions(-DUSE_OPENCL)
add_definitions(-DUSE_CLBLAST)
file(GLOB opencl "opencl/*.c")
set(libs ${libs} clblast)
endif()
endif()

if(USE_CLBLAS)
find_package(clBLAS REQUIRED)
if (clBLAS_FOUND)
add_definitions(-DUSE_OPENCL)
add_definitions(-DUSE_CLBLAS)
file(GLOB opencl "opencl/*.c")
set(libs ${libs} clBLAS)
endif()
endif()

if(USE_CUDA)
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
add_definitions(-DUSE_CUDA)
file(GLOB cuda "cuda/*.c")
include_directories(${CUDA_INCLUDE_DIRS})
set(libs ${libs} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()
endif()

if(USE_OPENCV)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
add_definitions(-DUSE_OPENCV)
set(libs ${libs} ${OpenCV_LIBS})
endif()
endif()

add_library(cnet ${core} ${layer} ${opencl} ${cuda})
target_link_libraries(cnet ${libs})

add_executable(linear_example example/linear_example.c)
target_link_libraries(linear_example cnet)

add_executable(mnist_example example/mnist_example.c example/mnist.c)
target_link_libraries(mnist_example cnet)

add_executable(xor_example example/xor_example.c)
target_link_libraries(xor_example cnet)

add_executable(vgg_example example/vgg_example.c)
target_link_libraries(vgg_example cnet)
