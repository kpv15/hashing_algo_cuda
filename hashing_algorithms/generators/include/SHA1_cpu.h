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
        uint8_t *u_word = (uint8_t *) word;
        unsigned long int calculatedWorkingBufferLength = calculateWorkingBufferLength();
        if (workingBuffer != nullptr && calculatedWorkingBufferLength != workingBufferLength)
            delete[] workingBuffer;
        if (workingBuffer == nullptr) {
            workingBuffer = new unsigned char[calculatedWorkingBufferLength];
            workingBufferLength = calculatedWorkingBufferLength;
            numberOfChunks = workingBufferLength / 64;
        }
            int i = 0, j, k = 0;
            while (i < defaultWordLength) {
                j = (i / 4) * 4 + 3 - (i % 4);
                workingBuffer[j] = word[i];
                i++;
            }
            j = (i / 4) * 4 + 3 - (i % 4);
            workingBuffer[j] = 0b10000000;
            i++;
            while (i < workingBufferLength - 2) {
                j = (i / 4) * 4 + 3 - (i % 4);
                workingBuffer[j] = 0b00000000;
                i++;
            }

            workingBuffer[j] = reinterpret_cast<uint32_t *>(&defaultWordLength)[k] * 8;
            uint32_t l = (defaultWordLength*8) % 0xFFFFFFFF;
            uint32_t h = (defaultWordLength*8) / 0xFFFFFFFF;;
            std::memcpy(workingBuffer + workingBufferLength - 4, &l, sizeof(uint32_t));
            std::memcpy(workingBuffer + workingBufferLength - 2, &h, sizeof(uint32_t));

    }

public:

    void setDefaultWordLength(unsigned int i)

    override;

    unsigned int getDigestLength()

    override;

    void calculateHashSum(uint8_t **digest, const char *word) {

        createWorkingBuffer(word);
        uint32_t w[80];

        block mdBuffer = DEFAULT_DIGEST_BUFFER;
        block stepBuffer;
        uint32_t temp;

        for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
            memcpy(w, workingBuffer + chunkNum * 16 * sizeof(uint32_t), 16 * sizeof(uint32_t));

            for (int i = 16; i <= 79; i++)
                w[i] = leftRotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

            stepBuffer = mdBuffer;

            for (int i = 0; i <= 79; i++) {
                if (i <= 19)
                    temp = leftRotate(stepBuffer.a, 5) + funF(i, stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                           stepBuffer.e + w[i] + 0x5A827999;
                else if (i <= 39)
                    temp = leftRotate(stepBuffer.a, 5) + funG(i, stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                           stepBuffer.e + w[i] + 0x6ED9EBA1;
                else if (i <= 59)
                    temp = leftRotate(stepBuffer.a, 5) + funH(i, stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                           stepBuffer.e + w[i] + 0x8F1BBCDC;
                else
                    temp = leftRotate(stepBuffer.a, 5) + funI(i, stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                           stepBuffer.e + w[i] + 0xCA62C1D6;
                stepBuffer.e = stepBuffer.d;
                stepBuffer.d = stepBuffer.c;
                stepBuffer.c = leftRotate(stepBuffer.b, 30);
                stepBuffer.b = stepBuffer.a;
                stepBuffer.a = temp;
            }
            mdBuffer.a += stepBuffer.a;
            mdBuffer.b += stepBuffer.b;
            mdBuffer.c += stepBuffer.c;
            mdBuffer.d += stepBuffer.d;
            mdBuffer.e += stepBuffer.e;
        }

        mdBuffer.a = __builtin_bswap32(mdBuffer.a);
        mdBuffer.b = __builtin_bswap32(mdBuffer.b);
        mdBuffer.c = __builtin_bswap32(mdBuffer.c);
        mdBuffer.d = __builtin_bswap32(mdBuffer.d);
        mdBuffer.e = __builtin_bswap32(mdBuffer.e);

        *digest = new unsigned char[DIGEST_LENGTH];
        memcpy(*digest, &mdBuffer, DIGEST_LENGTH);
    };

    virtual ~SHA1_cpu();

};


#endif //INYNIERKA_SHA1_CPU_H
