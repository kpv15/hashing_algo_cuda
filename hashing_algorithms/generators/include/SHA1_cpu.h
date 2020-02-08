//
// Created by grzegorz on 15.01.2020.
//

#ifndef INYNIERKA_SHA1_CPU_H
#define INYNIERKA_SHA1_CPU_H


#include <cstring>
#include "IHashingAlgorithm.h"

class SHA1_cpu : public IHashingAlgorithm {

    struct block {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
        uint32_t e;
    };

    const unsigned int DIGEST_LENGTH = 20;
    static const block DEFAULT_DIGEST_BUFFER;

    uint64_t defaultWordLength = 0;
    unsigned long int workingBufferLength = 0;
    unsigned int numberOfChunks = 0;
    unsigned char *workingBuffer = nullptr;

    uint32_t calculateWorkingBufferLength();

    uint32_t funF(const uint32_t b, const uint32_t c, const uint32_t d);

    uint32_t funG(const uint32_t b, const uint32_t c, const uint32_t d);

    uint32_t funH(const uint32_t b, const uint32_t c, const uint32_t d);

    uint32_t funI(const uint32_t b, const uint32_t c, const uint32_t d);

    uint32_t leftRotate(uint32_t x, unsigned int n);

    void createWorkingBuffer(const char *word);

public:

    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    void calculateHashSum(uint8_t **digest, const char *word) override;

    virtual ~SHA1_cpu();

};


#endif //INYNIERKA_SHA1_CPU_H
