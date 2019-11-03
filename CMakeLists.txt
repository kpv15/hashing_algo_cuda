cmake_minimum_required(VERSION 3.15)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(main add.cu dummy.cpp)

#clion cuda hack
include_directories(/opt/cuda/include)
