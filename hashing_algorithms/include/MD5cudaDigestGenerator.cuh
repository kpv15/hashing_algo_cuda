//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include "MD5_cuda.cuh"
#include "IGenerator.h"

class MD5cudaDigestGenerator : public IGenerator {
    char **words = nullptr;
    unsigned char **digest = nullptr;
    unsigned int n = 0;
    unsigned int length = 0;
    MD5_cuda md5Cuda;

public:
    void generate() override {

    }

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
