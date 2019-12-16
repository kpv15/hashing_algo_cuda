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
    static const unsigned int K[64];

    unsigned long int defaultWordLength = 0;
    unsigned long int workingBufferLength = 0;
    unsigned long int numberOfChunks = 0;
    unsigned char *workingBuffer = nullptr;
    block mdBuffer;

    unsigned int calculateWorkingBufferLength();

    void createWorkingBuffer(const char *word);

    unsigned int functionF(const unsigned int x, const unsigned int y, const unsigned int z);

    unsigned int functionG(const unsigned int x, const unsigned int y, const unsigned int z);

    unsigned int functionH(const unsigned int x, const unsigned int y, const unsigned int z);

    unsigned int functionI(const unsigned int x, const unsigned int y, const unsigned int z);

public:
    void setDefaultWordLength(unsigned int i) override;

    unsigned int getDigestLength() override;

    unsigned char *calculateHashSum(const char *word) override {
        createWorkingBuffer(word);
        mdBuffer = DEFAULT_DIGEST_BUFFER;

        unsigned int *chunks = reinterpret_cast<unsigned int *>(workingBuffer);

        for (unsigned int i = 0; i < numberOfChunks; i++) {
            block stepBuffer = mdBuffer;
            int F, g;
            for (int step = 0; step < 64; step++) {
                if (step < 16) {
                    F = functionF(st);
                    g = i;
                } else if (step < 32) {
                    F = functionG(stepBuffer);
                    g = (5 * i + 1) % 16;
                } else if (step < 48) {
                    F = functionH(stepBuffer);
                    g = (3 * i + 5) % 16;
                } else if (step < 64) {
                    F = functionI(stepBuffer);
                    g = (7 * i) % 16;
                }
                F += stepBuffer.a + K[i] + chunks[4 * i + g];
                stepBuffer.a = stepBuffer.d;
                stepBuffer.d = stepBuffer.c;
                stepBuffer.c = stepBuffer.b;
                stepBuffer.b += F << S[i];
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

    virtual ~MD5_cuda();
};


#endif //INYNIERKA_MD5_CUH
