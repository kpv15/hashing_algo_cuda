//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include "IGenerator.h"
#include "MD5_cuda.cuh"


class MD5cudaDigestGenerator : public IGenerator {

    static unsigned int calculateWorkingBufferLength(unsigned int defaultWordLength);

public:
    void generate() override;

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;

    bool needOneDimArray() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
