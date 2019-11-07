cmake_minimum_required(VERSION 3.15)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(main testAlg.cu dummy.cpp hashing_algorithms/include/IHashingAlgorithm.cuh hashing_algorithms/MD5.cu hashing_algorithms/include/MD5.cuh WordsGenerator.cpp include/WordsGenerator.h)

#clion cuda hack
include_directories(/opt/cuda/include)
