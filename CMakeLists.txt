cmake_minimum_required(VERSION 3.9)

project(cnet)

set(srcs ${srcs}
	"core/common.c"
	"core/data.c"
	"core/im2col.c"
	"core/layer.c"
	"core/net.c"
	"core/random.c"
	"layer/branch_layer.c"
	"layer/cee_layer.c"
	"layer/conv_layer.c"
	"layer/dropout_layer.c"
	"layer/fc_layer.c"
	"layer/merge_layer.c"
	"layer/mse_layer.c"
	"layer/normalization_layer.c"
	"layer/pooling_layer.c"
	"layer/relu_layer.c"
	"layer/scale_layer.c"
	"layer/sigmoid_layer.c"
	"layer/softmax_layer.c"
	"layer/tanh_layer.c"
)
include_directories(core layer)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
	add_definitions(-DUSE_CUDA)
	set(srcs ${srcs}
		"cuda/cudahelper.c"
		"cuda/gemm.c"
	)
	set(libs ${libs} CUDA::cublas)
	include_directories(cuda ${CUDAToolkit_INCLUDE_DIRS})
else(CUDAToolkit_FOUND)
	find_package(OpenCL)
	find_package(clBLAS)
if (OpenCL_FOUND AND clBLAS_FOUND)
	add_definitions(-DUSE_OPENCL)
	add_definitions(-DUSE_CLBLAS)
	set(srcs ${srcs}
		"opencl/clhelper.c"
		"opencl/clutil.c"
		"opencl/gemm.c"
	)
	set(libs ${libs} ${OpenCL_LIBRARIES} ${clBLAS_LIBRARIES})
	include_directories(opencl ${OpenCL_INCLUDE_DIRS} ${clBLAS_INCLUDE_DIRS})
else(OpenCL_FOUND AND clBLAS_FOUND)
	set(srcs ${srcs}
		"core/gemm.c"
	)
	find_package(BLAS)
if (BLAS_FOUND)
	add_definitions(-DUSE_BLAS)
	set(libs ${libs} BLAS::BLAS)
else (BLAS_FOUND)
	find_package(OpenMP)
if (OpenMP_FOUND)
	set(libs ${libs} OpenMP::OpenMP_C)
endif(OpenMP_FOUND)
endif(BLAS_FOUND)
endif(OpenCL_FOUND AND clBLAS_FOUND)
endif(CUDAToolkit_FOUND)

option(USE_OPENCV "use opencv as gui" OFF)
if(USE_OPENCV)
	find_package(OpenCV REQUIRED)
	add_definitions(-DUSE_OPENCV)
	set(libs ${libs} ${OpenCV_LIBS})
	include_directories(${OpenCV_INCLUDE_DIRS})
endif(USE_OPENCV)

option(USE_GFLAGS "use gflags" OFF)
if(USE_GFLAGS)
	find_package(GFlags REQUIRED)
	add_definitions(-DUSE_GFLAGS)
	set(srcs ${srcs}
		"gflags/flags.cc"
	)
	set(libs ${libs} gflags)
	include_directories(gflags)
endif(USE_GFLAGS)

if (UNIX)
	set(libs ${libs} m)
endif(UNIX)

add_library(cnet ${srcs})
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
