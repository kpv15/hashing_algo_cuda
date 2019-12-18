//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include "IGenerator.h"
#include "MD5_cuda.cuh"
#include <iostream>

class MD5cudaDigestGenerator : public IGenerator {

    static unsigned int calculateWorkingBufferLength(unsigned int defaultWordLength) {
        unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
        if (toAdd == 0) toAdd = 64;
        return defaultWordLength + toAdd + 8;
    }

public:
    void generate() override {
        unsigned char **digestGPU;
        unsigned char **wordsGPU;
        unsigned int workingBufferLength = calculateWorkingBufferLength(length_to_gen);
        unsigned int errorCode;
        if ((errorCode=cudaMalloc(&digestGPU, sizeof(unsigned char *) * n_to_gen)) != cudaSuccess) {
            std::cout << "error during alloc memory for digest on GPU error code: "<< errorCode << std::endl;
            return;
        };
        if ((errorCode=cudaMalloc(&wordsGPU, sizeof(unsigned char *) * n_to_gen)) != cudaSuccess) {
            std::cout << "error during alloc memory for words on GPU error code: "<< errorCode  << std::endl;
            return;
        };
        digest = new unsigned char*[n_to_gen];

        for (unsigned int i = 0; i < n_to_gen; i++) {
            if ((errorCode=cudaMalloc(&digestGPU[i], sizeof(unsigned char *) * getDigestLength()))!= cudaSuccess) {
                std::cout << "error during alloc memory for digest on GPU error code: "<< errorCode  << std::endl;
                return;
            };
            if ((errorCode=cudaMalloc(&wordsGPU[i], sizeof(unsigned char) * length_to_gen))!= cudaSuccess) {
                std::cout << "error during alloc memory for words on GPU error code: "<< errorCode  << std::endl;
                return;
            };
            cudaMemcpy(&wordsGPU[i], words[i], sizeof(unsigned char) * length_to_gen, cudaMemcpyHostToDevice);
            digest[i] = new unsigned char[length_to_gen];
        }

        calculateHashSum <<<1, n_to_gen>>> (digestGPU, words, workingBufferLength, length_to_gen);

        for (unsigned int i = 0; i < n_to_gen; i++) {
            cudaMemcpy(&digest, digestGPU, sizeof(unsigned char *) * length_to_gen, cudaMemcpyDeviceToHost);
            cudaFree(digestGPU[i]);
            cudaFree(wordsGPU[i]);
            delete[]digest[i];
        }
        cudaFree(digestGPU);
        cudaFree(wordsGPU);

        n = n_to_gen;
        length = length_to_gen;
    }

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
