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

#add_executable(MD5Iteration
#        kernelRun/iteration/MD5CrackerStarter.cu
#        kernelRun/iteration/MD5Kernel.cu
#        kernelRun/iteration/MD5KernelHeader.cuh
#        cudaUtils.cuh)
#
#add_executable(MD5Serial
#        kernelRun/serial/MD5CrackerStarter.cu
#        kernelRun/serial/MD5Kernel.cu
#        kernelRun/serial/MD5KernelHeader.cuh
#        cudaUtils.cuh)

add_executable(MD5Shared
        sharedMemory/MD5CrackerStarter.cu
        sharedMemory/MD5KernelHeader.cuh
        sharedMemory/MD5KernelShared.cu
        cudaUtils.cuh)

add_executable(MD5Private
        sharedMemory/MD5CrackerStarter.cu
        sharedMemory/MD5KernelHeader.cuh
        sharedMemory/MD5KernelPrivate.cu
        cudaUtils.cuh)

add_executable(MD5NonConst
        constMemory/nonConst/MD5CrackerStarter.cu
        constMemory/nonConst/MD5KernelHeader.cuh
        constMemory/nonConst/MD5Kernel.cu
        cudaUtils.cuh)

add_executable(MD5Const
        constMemory/const/MD5CrackerStarter.cu
        constMemory/const/MD5KernelHeader.cuh
        constMemory/const/MD5Kernel.cu
        cudaUtils.cuh)
set_target_properties(
        MD5Const
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if (CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(disarium PRIVATE "-Wall -Wextra")
endif ()