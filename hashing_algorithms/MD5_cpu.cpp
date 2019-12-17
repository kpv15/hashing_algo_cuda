//
// Created by grzegorz on 06.11.2019.
//

#include "include/MD5_cpu.h"

void MD5_cpu::setDefaultWordLength(unsigned int i) {
    this->defaultWordLength = i;
}

unsigned int MD5_cpu::getDigestLength() {
    return DIGEST_LENGTH;
}

unsigned int MD5_cpu::calculateWorkingBufferLength() {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return defaultWordLength + toAdd + 8;
}

void MD5_cpu::createWorkingBuffer(const char *word) {
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

const MD5_cpu::block MD5_cpu::DEFAULT_DIGEST_BUFFER = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476
};

const unsigned char MD5_cpu::S[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

const unsigned int MD5_cpu::T[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

const unsigned int MD5_cpu::K[64] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
        5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
        0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9
};

MD5_cpu::~MD5_cpu() {
    delete[] workingBuffer;
}

unsigned int MD5_cpu::funF(const unsigned int &x, const unsigned int &y, const unsigned int &z) {
    return (x & y) | ((~x) & z);
}

unsigned int MD5_cpu::funG(const unsigned int &x, const unsigned int &y, const unsigned int &z) {
    return (x & z) | (y & (~z));
}

unsigned int MD5_cpu::funH(const unsigned int &x, const unsigned int &y, const unsigned int &z) {
    return x ^ y ^ z;
}

unsigned int MD5_cpu::funI(const unsigned int &x, const unsigned int &y, const unsigned int &z) {
    return y ^ (x | (~z));
}

unsigned int MD5_cpu::leftRotate(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

unsigned char *MD5_cpu::calculateHashSum(const char *word) {
    createWorkingBuffer(word);
    block mdBuffer = DEFAULT_DIGEST_BUFFER;

    for (unsigned long i = 0; i < numberOfChunks; i++) {
        unsigned int *X = reinterpret_cast<unsigned int *>(workingBuffer + i * 16);

        block stepBuffer = mdBuffer;

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
}
