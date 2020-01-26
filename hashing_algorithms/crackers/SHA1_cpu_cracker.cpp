#include <cstring>
#include <cstdint>
#include "include/SHA1_cpu_cracker.h"

struct block {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
};

struct word {
    char d;
    char c;
    char b;
    char a;
};

const block DEFAULT_DIGEST_BUFFER = {
        0x67452301,
        0xEFCDAB89,
        0x98BADCFE,
        0x10325476,
        0xC3D2E1F0
};

uint32_t leftRotate(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

uint32_t funI(const uint32_t b, const uint32_t c, const uint32_t d) {
    return b ^ c ^ d;
}

uint32_t funH(const uint32_t b, const uint32_t c, const uint32_t d) {
    return (b & c) | (b & d) | (c & d);
}

uint32_t funG(const uint32_t b, const uint32_t c, const uint32_t d) {
    return b ^ c ^ d;
}

uint32_t funF(const uint32_t b, const uint32_t c, const uint32_t d) {
    return (b & c) | ((~b) & d);
}

uint32_t swap_bits(uint32_t x) {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(&x);
    return (ptr[3] << 0) | (ptr[2] << 8) | (ptr[1] << 16) | (ptr[0] << 24);
}

#define MAX_WORD_SIZE 10
#define MAX_WORKING_BUFFER_SIZE MAX_WORD_SIZE + 128

void calculateHashSum(unsigned char *digest_g, char *message, int workingBufferLength, int lenght) {

    uint32_t digest[DIGEST_LENGTH / 4];
    for (int i = 0; i < DIGEST_LENGTH / 4; i++)
        digest[i] = reinterpret_cast<uint32_t *>(digest_g)[i];

    unsigned char workingBuffer[MAX_WORKING_BUFFER_SIZE];
    //init working buffer
    workingBuffer[lenght] = 0b10000000;
    memset(workingBuffer + lenght + 1, 0, workingBufferLength * 4 - lenght - 1 - 8);

    uint64_t tmp = lenght * 8;
    memcpy(workingBuffer + workingBufferLength * 4 - 8, (uint32_t *) &tmp + 1, sizeof(uint32_t));
    memcpy(workingBuffer + workingBufferLength * 4 - 4, (uint32_t *) &tmp, sizeof(uint32_t));

    workingBuffer[0] = 'a';
    workingBuffer[1] = 'l';

    int combinations_outer = 1;
    int max_outer = (lenght - 2 < 2) ? lenght - 2 : 2;
    for (int i = 0; i < max_outer; i++)
        combinations_outer *= 256;
    int combinations_inner = 1;
    int max_inner = (lenght - 4 > 0) ? lenght - 4 : 0;
    for (int i = 0; i < max_inner; i++)
        combinations_inner *= 256;

    unsigned int numberOfChunks = workingBufferLength / 16;

    for (uint32_t k = 0; k < combinations_outer; k++) {
        reinterpret_cast<char *>(&workingBuffer)[0] = 'a';
        reinterpret_cast<char *>(&workingBuffer)[1] = 'l';
        memcpy(reinterpret_cast<char *>(&workingBuffer) + 2, &k, max_outer * sizeof(char));

        for (uint64_t j = 0; j < combinations_inner; j++) {

            memcpy(workingBuffer + 4, &j, max_inner * sizeof(char));
            uint32_t w[80];
            block mdBuffer = DEFAULT_DIGEST_BUFFER;
            block stepBuffer;
            uint32_t temp;

            for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
                for (int i = 0; i < 16; i++)
                    w[i] = swap_bits(reinterpret_cast<uint32_t *>(&workingBuffer + chunkNum * 16)[i]);
                if (chunkNum == numberOfChunks-1)
                    for (int i = 14; i < 16; i++)
                        w[i] = reinterpret_cast<uint32_t *>(&workingBuffer + chunkNum * 16)[i];

                for (int i = 16; i <= 79; i++)
                    w[i] = leftRotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

                stepBuffer = mdBuffer;

#pragma unroll
                for (int i = 0; i <= 79; i++) {
                    if (i <= 19)
                        temp = leftRotate(stepBuffer.a, 5) + funF(stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                               stepBuffer.e + w[i] + 0x5A827999;
                    else if (i <= 39)
                        temp = leftRotate(stepBuffer.a, 5) + funG(stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                               stepBuffer.e + w[i] + 0x6ED9EBA1;
                    else if (i <= 59)
                        temp = leftRotate(stepBuffer.a, 5) + funH(stepBuffer.b, stepBuffer.c, stepBuffer.d) +
                               stepBuffer.e + w[i] + 0x8F1BBCDC;
                    else
                        temp = leftRotate(stepBuffer.a, 5) + funI(stepBuffer.b, stepBuffer.c, stepBuffer.d) +
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

            if (mdBuffer.a == reinterpret_cast<uint32_t *>(digest)[0] &&
                mdBuffer.b == reinterpret_cast<uint32_t *>(digest)[1] &&
                mdBuffer.c == reinterpret_cast<uint32_t *>(digest)[2] &&
                mdBuffer.d == reinterpret_cast<uint32_t *>(digest)[3] &&
                mdBuffer.e == reinterpret_cast<uint32_t *>(digest)[4]) {
                memcpy(message, &workingBuffer, lenght * sizeof(char));
//            words[0] = '1';
//            words[1] = '2';
//            words[2] = '3';
//            words[3] = '4';
//            words[4] = '4';
            }
        }
    }
}
