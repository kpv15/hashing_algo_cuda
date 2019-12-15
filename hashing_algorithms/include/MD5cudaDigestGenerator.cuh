//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include <string>
#include "IGenerator.h"

class MD5cudaDigestGenerator: public IGenerator{
    void setN(unsigned int n) override {

    }

    void setLength(unsigned int length) override {

    }

    unsigned char **getDigits() override {
        return nullptr;
    }

    void setWords(char **words) override {

    }

    void generate() override {

    }

    unsigned int getDigestLength() override {
        return 0;
    }

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
