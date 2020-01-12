#include <cstring>
#include <cstdint>
#include "include/MD5_cuda.cuh"

#define DIGEST_LENGTH 16

struct block {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};

__constant__  const block DEFAULT_DIGEST_BUFFER = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476
};

__device__ unsigned int funF(const unsigned int x, const unsigned int y, const unsigned int z) {
    return (x & y) | ((~x) & z);
}

__device__ unsigned int funG(const unsigned int x, const unsigned int y, const unsigned int z) {
    return (x & z) | (y & (~z));
}

__device__ unsigned int funH(const unsigned int x, const unsigned int y, const unsigned int z) {
    return x ^ y ^ z;
}

__device__ unsigned int funI(const unsigned int x, const unsigned int y, const unsigned int z) {
    return y ^ (x | (~z));
}

__device__ unsigned int leftRotate(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

__global__ void calculateHashSum(unsigned char *digest, char *word, unsigned long int workingBufferLength,
                                 unsigned long int wordLength, unsigned long int n) {

    unsigned long int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < n) {

        unsigned char workingBuffer[1000];
        char word_cache[250];
        memcpy(word_cache, word + threadId* wordLength, wordLength);

        workingBuffer[wordLength] = 0b10000000;
        memset(workingBuffer + wordLength + 1, 0, workingBufferLength - wordLength - 1 - 8);
        reinterpret_cast<unsigned long *>(workingBuffer)[workingBufferLength / 8 - 1] = 8 * wordLength;

        unsigned int numberOfChunks = workingBufferLength / 64;

        block mdBuffer = DEFAULT_DIGEST_BUFFER;

        for (unsigned long i = 0; i < numberOfChunks; i++) {
            memcpy(workingBuffer, word_cache, wordLength);
            unsigned int X[16];
            memcpy(X, workingBuffer + i * 16 * sizeof(unsigned int), 16 * sizeof(unsigned int));
//            unsigned int *X = reinterpret_cast<unsigned int *>(workingBuffer + i * 16 * sizeof(unsigned int));

            block stepBuffer = mdBuffer;

            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funF(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[0] + 0xd76aa478), 7);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funF(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[1] + 0xe8c7b756), 12);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funF(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[2] + 0x242070db), 17);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funF(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[3] + 0xc1bdceee), 22);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funF(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[4] + 0xf57c0faf), 7);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funF(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[5] + 0x4787c62a), 12);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funF(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[6] + 0xa8304613), 17);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funF(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[7] + 0xfd469501), 22);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funF(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[8] + 0x698098d8), 7);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funF(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[9] + 0x8b44f7af), 12);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funF(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[10] + 0xffff5bb1), 17);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funF(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[11] + 0x895cd7be), 22);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funF(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[12] + 0x6b901122), 7);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funF(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[13] + 0xfd987193), 12);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funF(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[14] + 0xa679438e), 17);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funF(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[15] + 0x49b40821), 22);

            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funG(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[1] + 0xf61e2562), 5);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funG(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[6] + 0xc040b340), 9);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funG(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[11] + 0x265e5a51), 14);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funG(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[0] + 0xe9b6c7aa), 20);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funG(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[5] + 0xd62f105d), 5);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funG(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[10] + 0x02441453), 9);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funG(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[15] + 0xd8a1e681), 14);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funG(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[4] + 0xe7d3fbc8), 20);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funG(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[9] + 0x21e1cde6), 5);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funG(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[14] + 0xc33707d6), 9);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funG(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[3] + 0xf4d50d87), 14);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funG(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[8] + 0x455a14ed), 20);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funG(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[13] + 0xa9e3e905), 5);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funG(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[2] + 0xfcefa3f8), 9);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funG(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[7] + 0x676f02d9), 14);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funG(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[12] + 0x8d2a4c8a), 20);

            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funH(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[5] + 0xfffa3942), 4);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funH(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[8] + 0x8771f681), 11);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funH(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[11] + 0x6d9d6122), 16);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funH(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[14] + 0xfde5380c), 23);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funH(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[1] + 0xa4beea44), 4);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funH(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[4] + 0x4bdecfa9), 11);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funH(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[7] + 0xf6bb4b60), 16);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funH(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[10] + 0xbebfbc70), 23);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funH(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[13] + 0x289b7ec6), 4);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funH(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[0] + 0xeaa127fa), 11);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funH(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[3] + 0xd4ef3085), 16);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funH(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[6] + 0x04881d05), 23);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funH(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[9] + 0xd9d4d039), 4);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funH(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[12] + 0xe6db99e5), 11);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funH(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[15] + 0x1fa27cf8), 16);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funH(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[2] + 0xc4ac5665), 23);

            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funI(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[0] + 0xf4292244), 6);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funI(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[7] + 0x432aff97), 10);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funI(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[14] + 0xab9423a7), 15);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funI(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[5] + 0xfc93a039), 21);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funI(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[12] + 0x655b59c3), 6);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funI(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[3] + 0x8f0ccc92), 10);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funI(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[10] + 0xffeff47d), 15);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funI(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[1] + 0x85845dd1), 21);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funI(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[8] + 0x6fa87e4f), 6);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funI(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[15] + 0xfe2ce6e0), 10);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funI(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[6] + 0xa3014314), 15);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funI(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[13] + 0x4e0811a1), 21);
            stepBuffer.a = stepBuffer.b + leftRotate(
                    (stepBuffer.a + funI(stepBuffer.b, stepBuffer.c, stepBuffer.d) + X[4] + 0xf7537e82), 6);
            stepBuffer.d = stepBuffer.a + leftRotate(
                    (stepBuffer.d + funI(stepBuffer.a, stepBuffer.b, stepBuffer.c) + X[11] + 0xbd3af235), 10);
            stepBuffer.c = stepBuffer.d + leftRotate(
                    (stepBuffer.c + funI(stepBuffer.d, stepBuffer.a, stepBuffer.b) + X[2] + 0x2ad7d2bb), 15);
            stepBuffer.b = stepBuffer.c + leftRotate(
                    (stepBuffer.b + funI(stepBuffer.c, stepBuffer.d, stepBuffer.a) + X[9] + 0xeb86d391), 21);

            mdBuffer.a += stepBuffer.a;
            mdBuffer.b += stepBuffer.b;
            mdBuffer.c += stepBuffer.c;
            mdBuffer.d += stepBuffer.d;
        }
        memcpy((digest + threadId * DIGEST_LENGTH), &mdBuffer, DIGEST_LENGTH);
    }
}
