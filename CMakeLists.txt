cmake_minimum_required(VERSION 3.15)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenSSL REQUIRED)
add_executable(main
        testAlg.cu
        dummy.cpp
        hashing_algorithms/include/IHashingAlgorithm.cuh
        hashing_algorithms/MD5.cu
        hashing_algorithms/include/MD5.cuh
        WordsGenerator.cpp
        include/WordsGenerator.h
        hashing_algorithms/MD5_ssl.cpp
        hashing_algorithms/include/MD5_ssl.h
        include/HexParser.h)
target_link_libraries(main OpenSSL::SSL)

#clion cuda hack
include_directories(/opt/cuda/include)
