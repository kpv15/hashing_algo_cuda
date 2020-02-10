//
// Created by grzegorz on 12.01.2020.
//

#include <iostream>
#include <cstring>
#include <chrono>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/crackers/include/SHA1_cuda_cracker.cuh"

unsigned int calculateWorkingBufferLength(unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return (wordLength + toAdd + 8)/4;
}

int crack(int min_length, int max_length, unsigned char *digest);

inline unsigned char hexToInt(unsigned char a, unsigned char b) {
    a = a - '0' < 10 ? a - '0' : a - 'a' + 10;
    b = b - '0' < 10 ? b - '0' : b - 'a' + 10;
    return (a * 16) + b;
}

int main(int argc, char **argv) {

    char digest_hex[DIGEST_LENGTH * 2 + 1];
    unsigned char digest[DIGEST_LENGTH];
    int min = 0;
    int max = 0;
    if (argc >= 4) {
        min = atoi(argv[1]);
        max = atoi(argv[2]);
        strcpy(reinterpret_cast<char *>(&digest_hex), argv[3]);
    }

    for (int i = 0; i < DIGEST_LENGTH; i++) {
        digest[i] = (hexToInt(digest_hex[2 * i], digest_hex[2 * i + 1]));
    }

    for(int i = 0; i < DIGEST_LENGTH/4; i++){
        ((uint32_t*)&digest)[i]=__builtin_bswap32(((uint32_t*)&digest)[i]);
    }

    crack(min, max, digest);

}

int crack(int min_length, int max_length, unsigned char *digest) {

    min_length = min_length >= 2 ? min_length : 2;
    cudaError_t errorCode;
    const char NOT_FOUND[] = "-";

    char *word = new char[max_length + 1];
    char *word_gpu;

    unsigned char *digest_gpu;
    if ((errorCode = cudaMalloc((void **) &digest_gpu, DIGEST_LENGTH * sizeof(unsigned char))) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return 1;
    };

    cudaMemcpy(digest_gpu, digest, sizeof(unsigned char) * DIGEST_LENGTH, cudaMemcpyHostToDevice);

    bool *kernel_end_gpu;
    if ((errorCode = cudaMalloc((void **) &kernel_end_gpu, sizeof(bool))) != cudaSuccess) {
        std::cout << "error during alloc memory for kernel end flag on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return 1;
    };
    bool kerel_end = false;
    cudaMemcpy(kernel_end_gpu, &kerel_end, sizeof(bool), cudaMemcpyHostToDevice);

    for (int length = min_length; length <= max_length; length++) {
        if ((errorCode = cudaMalloc((void **) &word_gpu, length * sizeof(char))) != cudaSuccess) {
            std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                      << std::endl;
            return 1;
        };

        int workingBufferLength = calculateWorkingBufferLength(length);

        std::cout << "checking word with length: " << length << std::endl;
        auto startKernel = std::chrono::high_resolution_clock::now();

//        auto startKernel = std::chrono::high_resolution_clock::now();
        unsigned long threadNum = 128;
        unsigned long blockNum = 1;
        for (int i = 0; i < length && i < 4; ++i) {
            blockNum *= 256;
        }
        blockNum /= threadNum;

        calculateHashSum << < blockNum, threadNum >> > (digest_gpu, word_gpu, workingBufferLength, length,kernel_end_gpu);

        auto stopKernel = std::chrono::high_resolution_clock::now();

        if ((errorCode = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cout << "error during Device Synchronize: " << cudaGetErrorName(errorCode)
                      << std::endl;
            return 1;
        }
        cudaMemcpy(word, word_gpu, sizeof(char) * length, cudaMemcpyDeviceToHost);
        word[length] = '\0';

        auto durationKernel = std::chrono::duration_cast<std::chrono::microseconds>(stopKernel - startKernel);

        cudaMemcpy(&kerel_end, kernel_end_gpu, sizeof(bool), cudaMemcpyDeviceToHost);

        if(kerel_end)
            std::cout << word;
        else
            std::cout <<NOT_FOUND;

        std::cout  << "\tin: " << durationKernel.count() << std::endl;

        cudaFree(word_gpu);
    }

    cudaFree(digest_gpu);
    delete[]word;

    return 0;
}