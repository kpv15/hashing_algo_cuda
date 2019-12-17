//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CPU_H
#define INYNIERKA_MD5_CPU_H

#include <cstring>
#include "IHashingAlgorithm.cuh"

class MD5_cpu : public IHashingAlgorithm {

    struct block {
        unsigned int a;
        unsigned int b;
        unsigned int c;
        unsigned int d;
    };

    const unsigned int DIGEST_LENGTH = 16;
    static const unsigned char S[64];
    static const block DEFAULT_DIGEST_BUFFER;
    static const unsigned int T[64];
    static const unsigned int K[64];

    unsigned long int defaultWordLength = 0;
    unsigned long int workingBufferLength = 0;
    unsigned long int numberOfChunks = 0;
    unsigned char *workingBuffer = nullptr;

    unsigned int calculateWorkingBufferLength();

    void createWorkingBuffer(const char *word);

    unsigned int funF(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funG(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funH(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funI(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int leftRotate(unsigned int x, unsigned int n);

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override;

    virtual ~MD5_cpu();

};


#endif //INYNIERKA_MD5_CPU_H
