//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CPU_H
#define INYNIERKA_MD5_CPU_H

#include <cstring>
#include "IHashingAlgorithm.h"

class MD5_cpu : public IHashingAlgorithm {

    struct block {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
    };

    static const unsigned int DIGEST_LENGTH = 16;
    static const block DEFAULT_DIGEST_BUFFER;

    uint64_t defaultWordLength = 0;
    unsigned long int workingBufferLength = 0;
    unsigned long int numberOfChunks = 0;
    unsigned char *workingBuffer = nullptr;

    unsigned int calculateWorkingBufferLength();

    void createWorkingBuffer(const char *word);

    uint32_t funF(const uint32_t &x, const uint32_t &y, const uint32_t &z);

    uint32_t funG(const uint32_t &x, const uint32_t &y, const uint32_t &z);

    uint32_t funH(const uint32_t &x, const uint32_t &y, const uint32_t &z);

    uint32_t funI(const uint32_t &x, const uint32_t &y, const uint32_t &z);

    uint32_t leftRotate(uint32_t x, unsigned int n);

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    void calculateHashSum(unsigned char **digest, const char *word) override;

    virtual ~MD5_cpu();

};


#endif //INYNIERKA_MD5_CPU_H
