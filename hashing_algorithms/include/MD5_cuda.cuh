//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include "IHashingAlgorithm.cuh"

class MD5_cuda : public IHashingAlgorithm {

    const unsigned int DIGEST_LENGTH = 16;

    unsigned int defaultWordLength = 0;
    unsigned int workingBufferLength;

    static unsigned int calculateWorkingBufferLength(const unsigned int wordLength);

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override {
        return (unsigned char *) "Ȝ���xCH[9燠\u00012\u000F�av";
    };

};


#endif //INYNIERKA_MD5_CUH
