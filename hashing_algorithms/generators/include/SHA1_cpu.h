//
// Created by grzegorz on 15.01.2020.
//

#ifndef INYNIERKA_SHA1_CPU_H
#define INYNIERKA_SHA1_CPU_H


#include <cstring>
#include "IHashingAlgorithm.h"

class SHA1_cpu : public IHashingAlgorithm {

    struct block {
        unsigned int a;
        unsigned int b;
        unsigned int c;
        unsigned int d;
        unsigned int e;
    };

    const unsigned int DIGEST_LENGTH = 20;
    static const block DEFAULT_DIGEST_BUFFER;

    unsigned long int defaultWordLength = 0;
    unsigned long int workingBufferLength = 0;
    unsigned long int numberOfChunks = 0;
    unsigned char *workingBuffer = nullptr;

    uint32_t calculateWorkingBufferLength();

    uint32_t funF(const uint32_t t, const uint32_t b, const uint32_t c, const uint32_t d) {
        return (b & c) | ((~b) & d);
    }

    uint32_t funG(const uint32_t t, const uint32_t b, const uint32_t c, const uint32_t d) {
        return b ^ c ^ d;
    }

    uint32_t funH(const uint32_t t, const uint32_t b, const uint32_t c, const uint32_t d) {
        return (b & c) | (b & d) | (c & d);
    }

    uint32_t funI(const uint32_t t, const uint32_t b, const uint32_t c, const uint32_t d) {
        return b ^ c ^ d;
    }


    uint32_t leftRotate(uint32_t x, uint32_t n) {
        return (x << n) | (x >> (32 - n));
    }

    void createWorkingBuffer(const char *word) {
        unsigned long int calculatedWorkingBufferLength = calculateWorkingBufferLength();
        if (workingBuffer != nullptr && calculatedWorkingBufferLength != workingBufferLength)
            delete[] workingBuffer;
        if (workingBuffer == nullptr) {
            workingBuffer = new unsigned char[calculatedWorkingBufferLength];
            workingBufferLength = calculatedWorkingBufferLength;
            numberOfChunks = workingBufferLength / 64;
            workingBuffer[defaultWordLength] = 0b10000000;
            std::memset(workingBuffer + defaultWordLength + 1, 0, workingBufferLength - defaultWordLength - 1 - 8);
            reinterpret_cast<unsigned long *>(workingBuffer)[workingBufferLength / 8 - 1] = 8 * defaultWordLength;
        }
        std::memcpy(workingBuffer, word, defaultWordLength);
    }

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    void calculateHashSum(unsigned char **digest, const char *word) {

        createWorkingBuffer(word);
        uint32_t w[80];

        block mdBuffer = DEFAULT_DIGEST_BUFFER;
        block stepBuffer;
        uint temp;

        for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
            memcpy(w, workingBuffer + chunkNum * 16, 16);

            for (int i = 16; i < 79; i++)
                w[i] = leftRotate(w[i - 3] | w[i - 8] | w[i - 14] | w[i - 16], 1);

            stepBuffer = mdBuffer;

            for (int i = 0; i < 79; i++){
                if(i<=16);
                if(i<=39);
                if(i<=59);
                else;
            }

        }


    };

    virtual ~SHA1_cpu();

};


#endif //INYNIERKA_SHA1_CPU_H
