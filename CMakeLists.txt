cmake_minimum_required(VERSION 3.9)

project(cnet)

include_directories(core layer)

file(GLOB srcs "core/*.c" "layer/*.c")

add_library(cnet ${srcs})

add_executable(linear_sample sample/linear_sample.c)
target_link_libraries(linear_sample cnet m)

add_executable(mnist_sample sample/mnist_sample.c sample/mnist.c)
target_link_libraries(mnist_sample cnet m)

add_executable(xor_sample sample/xor_sample.c)
target_link_libraries(xor_sample cnet m)
