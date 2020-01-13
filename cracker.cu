//
// Created by grzegorz on 12.01.2020.
//

#include <iostream>
#include <cstring>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/MD5_cuda_cracker.cuh"

unsigned int calculateWorkingBufferLength(unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return wordLength + toAdd + 8;
}

int main(int argc, char **argv) {

    int length = 4;
    int workingBufferLength = calculateWorkingBufferLength(length);
    cudaError_t errorCode;

    unsigned char *digest = new unsigned char[DIGEST_LENGTH];
    reinterpret_cast<uint32_t*>(digest)[0] = 0x6fa64397;
    reinterpret_cast<uint32_t*>(digest)[1] = 0x49c24c91;
    reinterpret_cast<uint32_t*>(digest)[2] = 0x4416caef;
    reinterpret_cast<uint32_t*>(digest)[3] = 0x5c9ca185;

    unsigned char *digest_gpu;
    if ((errorCode = cudaMalloc((void **) &digest_gpu, DIGEST_LENGTH * sizeof(unsigned char))) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return 1;
    };


    char *word = new char[length + 1];
    strcpy(word,"----");
    char *word_gpu;
    if ((errorCode = cudaMalloc((void **) &word_gpu, length)) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return 1;
    };
    cudaMemcpy(digest, &digest_gpu, sizeof(unsigned char) * DIGEST_LENGTH, cudaMemcpyHostToDevice);

    calculateHashSum <<< 256, 256 >>> (digest_gpu, word_gpu, workingBufferLength, length);
    cudaDeviceSynchronize();

    cudaMemcpy(word, word_gpu, sizeof(char) * length, cudaMemcpyHostToDevice);
    word[length] = '\0';
    std::cout << word << std::endl;

    cudaFree(digest_gpu);
    cudaFree(word_gpu);
    delete[] digest;
}