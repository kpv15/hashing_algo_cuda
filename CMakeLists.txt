cmake_minimum_required(VERSION 3.10)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

set(UTILS_SOURCE
        utils/include/WordsGenerator.h
        utils/include/HexParser.h
        utils/include/ResultComparator.h
        utils/WordsGenerator.cpp
        utils/ResultComparator.cpp
)

add_executable(main
        main.cpp
        dummy.cpp
        hashing_algorithms/generators/include/IHashingAlgorithm.h
        hashing_algorithms/generators/MD5_cpu.cpp
        hashing_algorithms/generators/include/MD5_cpu.h
        hashing_algorithms/generators/MD5_ssl.cpp
        hashing_algorithms/generators/include/MD5_ssl.h
        hashing_algorithms/generators/MD5sslDigestGenerator.cpp
        hashing_algorithms/generators/include/MD5sslDigestGenerator.h
        hashing_algorithms/generators/include/HashingAlgorithms.h
        hashing_algorithms/generators/include/IGenerator.h
        hashing_algorithms/generators/MD5cudaDigestGenerator.cu
        hashing_algorithms/generators/include/MD5cudaDigestGenerator.cuh
        hashing_algorithms/generators/MD5_cuda.cu
        hashing_algorithms/generators/IGenerator.cpp
        hashing_algorithms/generators/include/MD5_cuda.cuh
        hashing_algorithms/generators/MD5cpuDigestGenerator.cpp
        hashing_algorithms/generators/include/MD5cpuDigestGenerator.h
        hashing_algorithms/generators/SHA1_cpu.cpp
        hashing_algorithms/generators/include/SHA1_cpu.h
        hashing_algorithms/generators/SHA1_ssl.cpp
        hashing_algorithms/generators/include/SHA1_ssl.h
        hashing_algorithms/generators/SHA1sslDigestGenerator.cpp
        hashing_algorithms/generators/include/SHA1sslDigestGenerator.h
        hashing_algorithms/generators/SHA1cpuDigestGenerator.cpp
        hashing_algorithms/generators/include/SHA1cpuDigestGenerator.h
        hashing_algorithms/generators/SHA1cudaDigestGenerator.cu
        hashing_algorithms/generators/include/SHA1cudaDigestGenerator.cuh
        hashing_algorithms/generators/SHA1_cuda.cu
        hashing_algorithms/generators/include/SHA1_cuda.cuh
        ${UTILS_SOURCE})

target_link_libraries(main OpenSSL::SSL)

add_executable(MD5cpuCrackerStarter
        MD5cpuCrackerStarter.cpp
        dummy.cpp
        hashing_algorithms/crackers/MD5_cpu_cracker.cpp
        hashing_algorithms/crackers/include/MD5_cpu_cracker.h
     )

add_executable(MD5sslCrackerStarter
        MD5sslCrackerStarter.cpp
        dummy.cpp
        )

target_link_libraries(MD5sslCrackerStarter OpenSSL::SSL)

add_executable(MD5cudaCrackerStarter
        MD5cudaCrackerStarter.cu
        dummy.cpp
        hashing_algorithms/crackers/MD5_cuda_cracker.cu
        hashing_algorithms/crackers/include/MD5_cuda_cracker.cuh
        )

#target_compile_options(MD5cudaCrackerStarter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        --maxrregcount=32
#        >)

add_executable(SHA1cudaCrackerStarter
        SHA1cudaCrackerStarter.cu
        dummy.cpp
        hashing_algorithms/crackers/include/SHA1_cuda_cracker.cuh
        hashing_algorithms/crackers/SHA1_cuda_cracker.cu
        )

add_executable(SHA1cpuCrackerStarter
        SHA1cpuCrackerStarter.cpp
        dummy.cpp
        hashing_algorithms/crackers/include/SHA1_cpu_cracker.h
        hashing_algorithms/crackers/SHA1_cpu_cracker.cpp
        )

add_executable(SHA1sslCrackerStarter
        SHA1sslCrackerStarter.cpp
        dummy.cpp
        )

target_link_libraries(SHA1sslCrackerStarter OpenSSL::SSL)

#clion cuda hack
include_directories(/opt/cuda/include)
include_directories(/opt/cuda/samples/common/inc)
