//
// Created by grzegorz on 20.01.2020.
//

#include "include/SHA1cudaDigestGenerator.cuh"
#include "include/SHA1_cuda.cuh"
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>

std::string SHA1cudaDigestGenerator::getAlgorithmName() {
    return "sha1_cuda";
}

unsigned int SHA1cudaDigestGenerator::getDigestLength() {
    return 20;
}

unsigned int SHA1cudaDigestGenerator::calculateWorkingBufferLength(unsigned int defaultWordLength) {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return (defaultWordLength + toAdd + 8)/4;
}

void SHA1cudaDigestGenerator::generate() {
    unsigned char *digestGPU;
    char *wordsGPU;

    auto startLoad = std::chrono::high_resolution_clock::now();

    unsigned long int workingBufferLength = calculateWorkingBufferLength(length_to_gen);
    if (workingBufferLength > 250) {
        std::cout << "error workingBufferLength > 2000 " << std::endl;
        return;
    }
    cudaError_t errorCode;

    if ((errorCode = cudaMalloc((void **) &digestGPU, sizeof(unsigned char) * n_to_gen * getDigestLength())) !=
        cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return;
    };

    unsigned int wordBufferLength = length_to_gen+4-length_to_gen%4;
    if ((errorCode = cudaMalloc(&wordsGPU, sizeof(char) * n_to_gen * wordBufferLength)) != cudaSuccess) {
        std::cout << "error during alloc memory for words on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return;
    };

    char *words_tmp = new char[wordBufferLength * n_to_gen];
    for (unsigned int i = 0; i < n_to_gen; i++) {
        memcpy(words_tmp + i * wordBufferLength, words[i], sizeof(unsigned char) * length_to_gen);
    }

    cudaMemcpy(wordsGPU, words_tmp, sizeof(unsigned char) * wordBufferLength * n_to_gen, cudaMemcpyHostToDevice);
    delete[] words_tmp;

    auto stopLoad = std::chrono::high_resolution_clock::now();
    auto durationLoad = std::chrono::duration_cast<std::chrono::milliseconds>(stopLoad - startLoad);
    std::cout << "gpu data load in: " << durationLoad.count() << " milliseconds" << std::endl;

    unsigned int blockSize = 128;
    unsigned int gridSize = (unsigned int) ceil((float) n_to_gen / blockSize);
    std::cout << "number of blocks: " << gridSize << "\t number of threads per block: " << blockSize << std::endl;

    auto startKernel = std::chrono::high_resolution_clock::now();

    SHA1_cuda::calculateHashSum <<< gridSize, blockSize >>> (digestGPU, wordsGPU, workingBufferLength, length_to_gen, n_to_gen);

    errorCode = cudaDeviceSynchronize();
    std::cout << "kernel quit code: " << cudaGetErrorName(errorCode) << std::endl;

    auto stopKernel = std::chrono::high_resolution_clock::now();
    auto durationKernel = std::chrono::duration_cast<std::chrono::milliseconds>(stopKernel - startKernel);
    std::cout << "kernel end work in in: " << durationKernel.count() << " milliseconds <-----------------" << std::endl;

    auto startUnload = std::chrono::high_resolution_clock::now();

    digest = new unsigned char *[n_to_gen];
    unsigned char *digest_tmp = new unsigned char[n_to_gen * getDigestLength()];
    cudaMemcpy(digest_tmp, digestGPU, sizeof(unsigned char) * getDigestLength() * n_to_gen, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < n_to_gen; i++) {
        digest[i] = new unsigned char[getDigestLength()];
        memcpy(digest[i], digest_tmp + i * getDigestLength(), getDigestLength());
    }

    delete[] digest_tmp;
    cudaFree(digestGPU);
    cudaFree(wordsGPU);
    auto stopUnload = std::chrono::high_resolution_clock::now();
    auto durationUnload = std::chrono::duration_cast<std::chrono::milliseconds>(stopUnload - startUnload);
    std::cout << "gpu data unload in: " << durationLoad.count() << " milliseconds" << std::endl;

    n = n_to_gen;
    length = length_to_gen;
}
