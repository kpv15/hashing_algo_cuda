//
// Created by grzegorz on 15.12.2019.
//


#include "include/MD5cudaDigestGenerator.cuh"
#include <iostream>
#include <chrono>

std::string MD5cudaDigestGenerator::getAlgorithmName() {
    return "md5_cuda";
}

unsigned int MD5cudaDigestGenerator::getDigestLength() {
    return 16;
}

void MD5cudaDigestGenerator::generate() {
    unsigned char *digestGPU;
    char *wordsGPU;

    auto startLoad = std::chrono::high_resolution_clock::now();

    unsigned long int workingBufferLength = calculateWorkingBufferLength(length_to_gen);
    cudaError_t errorCode;

    if ((errorCode = cudaMalloc((void **) &digestGPU, sizeof(unsigned char) * n_to_gen * getDigestLength())) != cudaSuccess) {
        std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return;
    };
    if ((errorCode = cudaMalloc(&wordsGPU, sizeof(char) * n_to_gen * length_to_gen)) != cudaSuccess) {
        std::cout << "error during alloc memory for words on GPU error code: " << cudaGetErrorName(errorCode)
                  << std::endl;
        return;
    };

    for (unsigned int i = 0; i < n_to_gen; i++) {
        cudaMemcpy(wordsGPU+i*length_to_gen, words[i], sizeof(unsigned char) * length_to_gen, cudaMemcpyHostToDevice);
    }

    auto stopLoad = std::chrono::high_resolution_clock::now();
    auto durationLoad = std::chrono::duration_cast<std::chrono::milliseconds>(stopLoad - startLoad);
    std::cout << "gpu data load in: " << durationLoad.count() << " milliseconds" << std::endl;

    auto startKernel = std::chrono::high_resolution_clock::now();

    calculateHashSum <<< 1, 1024 >>> (digestGPU, wordsGPU, workingBufferLength, length_to_gen, n_to_gen);

    cudaDeviceSynchronize();

    auto stopKernel = std::chrono::high_resolution_clock::now();
    auto durationKernel = std::chrono::duration_cast<std::chrono::milliseconds>(stopKernel - startKernel);
    std::cout << "kernel end work in in: " << durationKernel.count() << " milliseconds" << std::endl;

    auto startUnload = std::chrono::high_resolution_clock::now();

    digest = new unsigned char *[n_to_gen];
    for (unsigned int i = 0; i < n_to_gen; i++) {
        digest[i] = new unsigned char[ getDigestLength()];
        cudaMemcpy(digest[i], digestGPU+i*getDigestLength(), sizeof(unsigned char) * getDigestLength(), cudaMemcpyDeviceToHost);
    }
    cudaFree(digestGPU);
    cudaFree(wordsGPU);
    auto stopUnload = std::chrono::high_resolution_clock::now();
    auto durationUnload = std::chrono::duration_cast<std::chrono::milliseconds>(stopUnload - startUnload);
    std::cout << "gpu data unload in: " << durationLoad.count() << " milliseconds" << std::endl;

    n = n_to_gen;
    length = length_to_gen;
}

unsigned int MD5cudaDigestGenerator::calculateWorkingBufferLength(unsigned int defaultWordLength) {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return defaultWordLength + toAdd + 8;
}
