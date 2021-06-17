#include <cstring>
#include <cstdint>
#include "MD5KernelHeader.cuh"

struct block {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};

__constant__ unsigned char DIGEST[DIGEST_LENGTH];
__constant__ int WORKING_BUFFER_LENGTH;
__constant__ int LENGTH;

__constant__  const block DEFAULT_DIGEST_BUFFER = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476
};

__constant__ const unsigned char S[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__constant__ const uint32_t T[64] = {
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

__constant__ const uint32_t K[64] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
        5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
        0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9
};

__device__ unsigned int funF(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (x & y) | ((~x) & z);
}

__device__ unsigned int funG(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (x & z) | (y & (~z));
}

__device__ unsigned int funH(const uint32_t x, const uint32_t y, const uint32_t z) {
    return x ^ y ^ z;
}

__device__ unsigned int funI(const uint32_t x, const uint32_t y, const uint32_t z) {
    return y ^ (x | (~z));
}

__device__ unsigned int leftRotate(uint32_t x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

#define MAX_WORD_SIZE 10
#define MAX_WORKING_BUFFER_SIZE MAX_WORD_SIZE + 128

__global__ void
calculateHashSum(char *word, volatile bool *kernel_end) {

    __shared__ bool done;
    __shared__ uint32_t workingBuffer[MAX_WORKING_BUFFER_SIZE / 4];
    //init working buffer
    if (threadIdx.x == 0) {
        memset(workingBuffer, 0, WORKING_BUFFER_LENGTH);
        reinterpret_cast<uint8_t *>(workingBuffer)[LENGTH] = 0b10000000;
        unsigned long tmp = 8 * LENGTH;
        memcpy(workingBuffer + WORKING_BUFFER_LENGTH / 4 - 2, &tmp, sizeof(uint64_t));
        done = false;
    }
    __syncthreads();

    unsigned int numberOfChunks = WORKING_BUFFER_LENGTH / 64;
    do {
        block mdBuffer = DEFAULT_DIGEST_BUFFER;

        for (unsigned long i = 0; i < numberOfChunks; i++) {
            unsigned int *X = workingBuffer + i * 16;

            block stepBuffer = mdBuffer;

            uint32_t *a = &stepBuffer.a;
            uint32_t *b = &stepBuffer.b;
            uint32_t *c = &stepBuffer.c;
            uint32_t *d = &stepBuffer.d;
            uint32_t *tmp;

#pragma unroll
            for (unsigned int step = 0; step < 64; step++) {
                unsigned int xStep = K[step] != 0 ? X[K[step]] : workingBuffer[0] | (blockIdx.x * blockDim.x) | threadIdx.x;
                if (step < 16) {
                    *a = *b + leftRotate((*a + funF(*b, *c, *d) + xStep + T[step]), S[step]);
                } else if (step < 32) {
                    *a = *b + leftRotate((*a + funG(*b, *c, *d) + xStep + T[step]), S[step]);
                } else if (step < 48) {
                    *a = *b + leftRotate((*a + funH(*b, *c, *d) + xStep + T[step]), S[step]);
                } else {
                    *a = *b + leftRotate((*a + funI(*b, *c, *d) + xStep + T[step]), S[step]);
                }

                tmp = d;
                d = c;
                c = b;
                b = a;
                a = tmp;
            }

            mdBuffer.a += *a;
            mdBuffer.b += *b;
            mdBuffer.c += *c;
            mdBuffer.d += *d;
        }
        if (mdBuffer.a == reinterpret_cast<uint32_t *>(DIGEST)[0] &&
            mdBuffer.b == reinterpret_cast<uint32_t *>(DIGEST)[1] &&
            mdBuffer.c == reinterpret_cast<uint32_t *>(DIGEST)[2] &&
            mdBuffer.d == reinterpret_cast<uint32_t *>(DIGEST)[3]) {

            memcpy(word, workingBuffer, LENGTH * sizeof(char));
            reinterpret_cast<uint32_t *>(word)[0] += (blockIdx.x * blockDim.x) | threadIdx.x;
            *kernel_end = true;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            unsigned char *tmp = reinterpret_cast<unsigned char *>(workingBuffer);
            if (threadIdx.x == 0) {
                int i = 4;
                do {
                    tmp[i++]++;
                } while (tmp[i] == 0 && i < LENGTH);
                done = true;
                for (int i = 4; i < LENGTH; i++) {
                    if (tmp[i] != 0) {
                        done = false;
                    }
                }
            }
        }
        __syncthreads();

    } while (!(done || *kernel_end));

}
