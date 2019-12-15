//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include <cstring>
#include "IHashingAlgorithm.cuh"

class MD5_cuda : public IHashingAlgorithm {

    const unsigned int DIGEST_LENGTH = 16;

    unsigned int defaultWordLength = 0;
    unsigned int workingBufferLength;
    unsigned char *workingBuffer = nullptr;

    unsigned int calculateWorkingBufferLength();

    void createWorkingBuffer(const char *word) {
        if (workingBuffer != nullptr)
            delete[] workingBuffer;
        workingBufferLength = calculateWorkingBufferLength();
        workingBuffer = new unsigned char[workingBufferLength];
        std::memcpy(workingBuffer, word, defaultWordLength);
        workingBuffer[defaultWordLength] = 0b10000000;
        std::memset(workingBuffer + defaultWordLength + 1, 0, workingBufferLength - defaultWordLength -1);
    };

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override {
        createWorkingBuffer(word);



        unsigned char *toReturn = workingBuffer;
        workingBuffer = nullptr;
        return toReturn;
    };

};


#endif //INYNIERKA_MD5_CUH
