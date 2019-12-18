//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include "IGenerator.h"
#include "MD5_cuda.cuh"
#include <iostream>
#include <cstring>

class MD5cudaDigestGenerator : public IGenerator {

    static unsigned int calculateWorkingBufferLength(unsigned int defaultWordLength) {
        unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
        if (toAdd == 0) toAdd = 64;
        return defaultWordLength + toAdd + 8;
    }

public:
    void generate() override {
        unsigned char *digestGPU;
        char *wordsGPU;
        unsigned int workingBufferLength = calculateWorkingBufferLength(length_to_gen);
        cudaError_t errorCode;

        digestGPU= static_cast<unsigned char *>(malloc(sizeof(unsigned char) * n_to_gen * getDigestLength()));
//        if ((errorCode = cudaMalloc((void **) &digestGPU, sizeof(char) * n_to_gen * getDigestLength())) !=
//            cudaSuccess) {
//            std::cout << "error during alloc memory for digest on GPU error code: " << cudaGetErrorName(errorCode)
//                      << std::endl;
//            return;
//        };
        wordsGPU= static_cast<char *>(malloc(sizeof(char) * n_to_gen * length_to_gen));
//        if ((errorCode = cudaMalloc(&wordsGPU, sizeof(unsigned char *) * n_to_gen * length_to_gen)) != cudaSuccess) {
//            std::cout << "error during alloc memory for words on GPU error code: " << cudaGetErrorName(errorCode)
//                      << std::endl;
//            return;
//        };

        for (unsigned int i = 0; i < n_to_gen; i++) {
//            cudaMemcpy(wordsGPU+i*length_to_gen, words[i], sizeof(unsigned char) * length_to_gen, cudaMemcpyHostToDevice);
                     memcpy(wordsGPU + i * length_to_gen, words[i], sizeof(char) * length_to_gen);
        }

        for(unsigned int i =0;i<n_to_gen;i++)
            calculateHashSum(digestGPU, wordsGPU, workingBufferLength, length_to_gen);
//        calculateHashSum <<< 1, n_to_gen >>> (digestGPU, wordsGPU, workingBufferLength, length_to_gen);

//        cudaDeviceSynchronize();

        digest = new unsigned char *[n_to_gen];
        for (unsigned int i = 0; i < n_to_gen; i++) {
            digest[i] = new unsigned char[ getDigestLength()];
//            cudaMemcpy(digest[i], digestGPU+i*getDigestLength(), sizeof(unsigned char *) * getDigestLength(), cudaMemcpyDeviceToHost);
            memcpy(digest[i], digestGPU+i*getDigestLength(), sizeof(unsigned char) * getDigestLength());
        }
//        cudaFree(digestGPU);
//        cudaFree(wordsGPU);
        free(digestGPU);
        free(wordsGPU);

        n = n_to_gen;
        length = length_to_gen;
    }

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
