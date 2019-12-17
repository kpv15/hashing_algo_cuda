//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include <cstring>
#include "IHashingAlgorithm.cuh"

class MD5_cuda : public IHashingAlgorithm {

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
    block mdBuffer;

    unsigned int calculateWorkingBufferLength();

    void createWorkingBuffer(const char *word);

    unsigned int funF(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funG(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funH(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int funI(const unsigned int &x, const unsigned int &y, const unsigned int &z);

    unsigned int leftRotate(unsigned int x, unsigned int n) {
        return (x << n) | (n >> (32 - n));
    }

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override {
        createWorkingBuffer(word);
        mdBuffer = DEFAULT_DIGEST_BUFFER;
        block stepBuffer;

        for (unsigned int i = 0; i < numberOfChunks; i++) {
            unsigned int *X = reinterpret_cast<unsigned int *>(workingBuffer + i * 16);

            stepBuffer = mdBuffer;

            unsigned int *a = &stepBuffer.a, *b = &stepBuffer.b, *c = &stepBuffer.c, *d = &stepBuffer.d, *tmp;

            for (unsigned int step = 0; step < 64; step++) {
                if (step < 16) {
                    *a = *b + leftRotate((*a + funF(*b, *c, *d) + X[K[step]] + T[step]), S[step]);
                } else if (step < 32) {
                    *a = *b + leftRotate((*a + funG(*b, *c, *d) + X[K[step]] + T[step]), S[step]);
                } else if (step < 48) {
                    *a = *b + leftRotate((*a + funH(*b, *c, *d) + X[K[step]] + T[step]), S[step]);
                } else {
                    *a = *b + leftRotate((*a + funI(*b, *c, *d) + X[K[step]] + T[step]), S[step]);
                }

                tmp = d;
                d = c;
                c = b;
                b = a;
                a = tmp;
            }

            mdBuffer.a += stepBuffer.a;
            mdBuffer.b += stepBuffer.b;
            mdBuffer.c += stepBuffer.c;
            mdBuffer.d += stepBuffer.d;
        }

        unsigned char *toReturn = new unsigned char[DIGEST_LENGTH];
        memcpy(toReturn, &mdBuffer, DIGEST_LENGTH);
        return toReturn;
    };

    virtual ~

    MD5_cuda();

};


#endif //INYNIERKA_MD5_CUH
