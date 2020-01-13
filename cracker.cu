//
// Created by grzegorz on 12.01.2020.
//

#include <iostream>
#include <cstring>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/MD5_cuda_cracker.cuh"
//#include "hashing_algorithms/include/MD5_cpu_cracker.h"

unsigned int calculateWorkingBufferLength(unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return wordLength + toAdd + 8;
}

int crack(int min_length, int max_length);

int main(int argc, char **argv) {

    int min = 0;
    int max = 0;
    if (argc >= 3) {
        min = atoi(argv[1]);
        max = atoi(argv[2]);
    }
    crack(min, max);
}

int crack(int min_length, int max_length) {

    min_length = min_length >= 2 ? min_length : 2;
    cudaError_t errorCode;
    const char NOT_FOUND[] = "-";

    char *word =  new char[max_length + 1];
    char *word_gpu;

    unsigned char *digest = new unsigned char[DIGEST_LENGTH];
    unsigned char *digest_gpu;
    if ((errorCode = cudaMalloc((void **) &digest_gpu, DIGEST_LENGTH * sizeof(unsigned char))) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return 1;
    };

    reinterpret_cast<uint32_t *>(digest)[0] = 0xd5e1682c;
    reinterpret_cast<uint32_t *>(digest)[1] = 0xaee40908;
    reinterpret_cast<uint32_t *>(digest)[2] = 0xfecf7b35;
    reinterpret_cast<uint32_t *>(digest)[3] = 0x2a9dc91f;
    cudaMemcpy(digest_gpu, digest, sizeof(unsigned char) * DIGEST_LENGTH, cudaMemcpyHostToDevice);

    for (int length = min_length; length <= max_length; length++) {
        if ((errorCode = cudaMalloc((void **) &word_gpu, length * sizeof(char))) != cudaSuccess) {
            std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                      << std::endl;
            return 1;
        };
        cudaMemcpy(word_gpu, NOT_FOUND, sizeof(char) * (strlen(NOT_FOUND) + 1), cudaMemcpyHostToDevice);

        int workingBufferLength = calculateWorkingBufferLength(length);

        std::cout << "checking word with length: " << length << std::endl;

        calculateHashSum << < 256, 256 >> > (digest_gpu, word_gpu, workingBufferLength, length);

        cudaDeviceSynchronize();
        cudaMemcpy(word, word_gpu, sizeof(char) * length, cudaMemcpyDeviceToHost);
        word[length] = '\0';
        std::cout << word << std::endl;

        cudaFree(word_gpu);
    }

    cudaFree(digest_gpu);
    delete[] digest;
    delete[]word;

    return 0;
}