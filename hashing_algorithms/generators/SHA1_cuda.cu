//
// Created by grzegorz on 20.01.2020.
//

#include <cstring>
#include <cstdint>
#include "include/SHA1_cuda.cuh"

#define DIGEST_LENGTH 20
namespace SHA1_cuda {
    struct block {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
        uint32_t e;
    };

    __constant__ block DEFAULT_DIGEST_BUFFER = {
            0x67452301,
            0xEFCDAB89,
            0x98BADCFE,
            0x10325476,
            0xC3D2E1F0
    };

    __device__ uint32_t leftRotate(uint32_t x, uint32_t n) {
        return (x << n) | (x >> (32 - n));
    }

    __device__ uint32_t funI(const uint32_t b, const uint32_t c, const uint32_t d) {
        return b ^ c ^ d;
    }

    __device__ uint32_t funH(const uint32_t b, const uint32_t c, const uint32_t d) {
        return (b & c) | (b & d) | (c & d);
    }

    __device__ uint32_t funG(const uint32_t b, const uint32_t c, const uint32_t d) {
        return b ^ c ^ d;
    }

    __device__ uint32_t funF(const uint32_t b, const uint32_t c, const uint32_t d) {
        return (b & c) | ((~b) & d);
    }

    __device__ uint32_t swap_bits(uint32_t x) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(&x);
        return (ptr[3] << 0) | (ptr[2] << 8) | (ptr[1] << 16) | (ptr[0] << 24);
    }

    __device__ void fillWorkingBuffer(const char *word, uint32_t *workingBuffer, unsigned int workingBufferLength,
                                      unsigned int wordLength) {
        unsigned int i = 0, j;
        uint32_t *word_ptr = (uint32_t *) word;
        for (i = 0; i < wordLength / 4; i++)
            workingBuffer[i] = swap_bits(word_ptr[i]);

        uint32_t split_word = 0;
        for(j = 0; j < wordLength%4 ;j++)
            ((uint8_t *)&split_word)[3-j]=word[wordLength/4*4+j];
        ((uint8_t *)&split_word)[3-j]=0b10000000;

        workingBuffer[i] = split_word;
        i++;

        while (i < workingBufferLength - 2) {
            workingBuffer[i++] = 0;
        }

        uint64_t tmp = wordLength * 8;
        std::memcpy(workingBuffer + i++, (uint32_t *) &tmp + 1, sizeof(uint32_t));
        std::memcpy(workingBuffer + i++, (uint32_t *) &tmp, sizeof(uint32_t));

    }

    __global__ void calculateHashSum(unsigned char *digest, const char *word, unsigned long int workingBufferLength,
                                     unsigned long int wordLength, unsigned long int n) {

        unsigned long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int wordBufferLength = wordLength+4-wordLength%4;

        if (threadId < n) {

            uint32_t workingBuffer[150];

            fillWorkingBuffer(word + wordBufferLength * threadId, workingBuffer, workingBufferLength, wordLength);

            uint32_t w[80];
            unsigned int numberOfChunks = workingBufferLength / 16;

            block mdBuffer = DEFAULT_DIGEST_BUFFER;
            block stepBuffer;
            uint32_t temp;

            for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
                memcpy(w, workingBuffer + chunkNum * 16, 16 * sizeof(uint32_t));

                for (int i = 16; i <= 79; i++)
                    w[i] = leftRotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

                stepBuffer = mdBuffer;

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

            mdBuffer.a = swap_bits(mdBuffer.a);
            mdBuffer.b = swap_bits(mdBuffer.b);
            mdBuffer.c = swap_bits(mdBuffer.c);
            mdBuffer.d = swap_bits(mdBuffer.d);
            mdBuffer.e = swap_bits(mdBuffer.e);

//            mdBuffer.a=mdBuffer.b=mdBuffer.c=mdBuffer.d=mdBuffer.e = 0;
            memcpy(digest + threadId * DIGEST_LENGTH, &mdBuffer, DIGEST_LENGTH);

        }

    }

}
