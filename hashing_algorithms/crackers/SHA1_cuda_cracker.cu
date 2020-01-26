#include <cstring>
#include <cstdint>
#include "include/SHA1_cuda_cracker.cuh"

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

#define MAX_WORD_SIZE 10
#define MAX_WORKING_BUFFER_SIZE MAX_WORD_SIZE + 128

__global__ void calculateHashSum(unsigned char *digest_g, char *message, int workingBufferLength, int lenght) {
    __shared__ uint32_t digest[DIGEST_LENGTH / 4];
    for (int i = threadIdx.x; i < DIGEST_LENGTH / 4; i += blockDim.x)
        digest[i] = reinterpret_cast<uint32_t *>(digest_g)[i];
    __syncthreads();

    unsigned char workingBuffer[MAX_WORKING_BUFFER_SIZE];
    memset(workingBuffer, 0, workingBufferLength * 4);
    //init working buffer
    workingBuffer[lenght] = 0b10000000;

    uint64_t tmp = lenght * 8;
    uint32_t l = swap_bits(((uint32_t *) &tmp)[0]);
    uint32_t h = swap_bits(((uint32_t *) &tmp)[1]);
    memcpy(workingBuffer + workingBufferLength * 4 - 8, &h, sizeof(uint32_t));
    memcpy(workingBuffer + workingBufferLength * 4 - 4, &l, sizeof(uint32_t));

    workingBuffer[0] = blockIdx.x;
    workingBuffer[1] = threadIdx.x;

    unsigned int numberOfChunks = workingBufferLength / 16;

    bool done;
    do {
        uint32_t w[80];
        block mdBuffer = DEFAULT_DIGEST_BUFFER;
        block stepBuffer;
        uint32_t temp;

        for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
#pragma unroll
            for (int i = 0; i < 16; i++)
                w[i] = swap_bits(reinterpret_cast<uint32_t *>(&workingBuffer + chunkNum * 16)[i]);

#pragma unroll
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
            break;
        }
        int i = 2;
        while (++workingBuffer[i] == 0 && i < lenght)
            i++;
        done = true;
        for (int i = 2; i < lenght; i++) {
            if (workingBuffer[i] != 0) {
                done = false;
                break;
            }
        }
    } while (!done);
}