cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(inynierka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

#loop tests
set(LOOP_TEST_FRAMEWORK
        loopTests/MD5cudaCrackerStarter.cu
        cudaUtils.cuh
        loopTests/include/MD5KernelHeader.cuh
        )

add_executable(MD5StepsList
        ${LOOP_TEST_FRAMEWORK}
        loopTests/MD5StepsList.cu
        )

add_executable(MD5PragmaLoop
        ${LOOP_TEST_FRAMEWORK}
        loopTests/MD5PragmaLoop.cu
        )

add_executable(MD5Loop
        ${LOOP_TEST_FRAMEWORK}
        loopTests/MD5Loop.cu
        )

if (CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(disarium PRIVATE "-Wall -Wextra")
endif ()