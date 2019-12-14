//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include "IHashingAlgorithm.cuh"

    class MD5_cuda : public IHashingAlgorithm {

    const unsigned int digestLength = 16;

    unsigned int defaultWordLength = 0;
    unsigned int workingLength;

    static unsigned int calculateWorkingLength(const unsigned int wordLenth);

public:
    std::string calculateHashSum(std::string word) override{};

    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override{};

};


#endif //INYNIERKA_MD5_CUH
