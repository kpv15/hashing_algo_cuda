cmake_minimum_required(VERSION 3.15)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

set(UTILS_SOURCE
        utils/include/WordsGenerator.h
        utils/include/HexParser.h
        utils/include/ResultComparator.h
        utils/WordsGenerator.cpp
        utils/ResultComparator.cpp)

add_executable(main
        main.cu
        dummy.cpp
        hashing_algorithms/include/IHashingAlgorithm.cuh
        hashing_algorithms/MD5_cpu.cpp
        hashing_algorithms/include/MD5_cpu.h
        hashing_algorithms/MD5_ssl.cpp
        hashing_algorithms/include/MD5_ssl.h
        hashing_algorithms/MD5sslDigestGenerator.cpp
        hashing_algorithms/include/MD5sslDigestGenerator.h
        hashing_algorithms/include/HashingAlgorithms.h
        hashing_algorithms/include/IGenerator.h
        hashing_algorithms/MD5cudaDigestGenerator.cu
        hashing_algorithms/include/MD5cudaDigestGenerator.cuh
        hashing_algorithms/include/MD5_cuda.cuh
        hashing_algorithms/MD5_cuda.cu
        hashing_algorithms/IGenerator.cpp
        ${UTILS_SOURCE})

target_link_libraries(main OpenSSL::SSL)

add_executable(tests
        tests/tests.cpp
        dummy.cpp
        hashing_algorithms/include/IHashingAlgorithm.cuh
        hashing_algorithms/MD5_cpu.cpp
        hashing_algorithms/include/MD5_cpu.h
        hashing_algorithms/include/MD5_cuda.cuh
        hashing_algorithms/MD5_cuda.cu)

#clion cuda hack
include_directories(/opt/cuda/include)
