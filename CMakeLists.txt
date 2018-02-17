cmake_minimum_required(VERSION 3.9)

project(cnet)

include_directories(core layer)

file(GLOB srcs "core/*.c" "layer/*.c")

add_library(cnet ${srcs})

add_executable(linear_example example/linear_example.c)
target_link_libraries(linear_example cnet m)

add_executable(mnist_example example/mnist_example.c example/mnist.c)
target_link_libraries(mnist_example cnet m)

add_executable(xor_example example/xor_example.c)
target_link_libraries(xor_example cnet m)
