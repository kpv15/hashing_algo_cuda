//
// Created by grzegorz on 20.01.2020.
//

#ifndef INYNIERKA_SHA1CUDADIGESTGENERATOR_CUH
#define INYNIERKA_SHA1CUDADIGESTGENERATOR_CUH


#include "IGenerator.h"


class SHA1cudaDigestGenerator: public IGenerator {
    static unsigned int calculateWorkingBufferLength(unsigned int defaultWordLength);

public:
    void generate();

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_SHA1CUDADIGESTGENERATOR_CUH
