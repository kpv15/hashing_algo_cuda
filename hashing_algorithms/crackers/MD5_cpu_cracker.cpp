#include <cstring>
#include <cstdint>
#include "include/MD5_cpu_cracker.h"
#include "../../cuda_clion_hack.hpp"

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

unsigned int funF(const unsigned int x, const unsigned int y, const unsigned int z) {
    return (x & y) | ((~x) & z);
}

unsigned int funG(const unsigned int x, const unsigned int y, const unsigned int z) {
    return (x & z) | (y & (~z));
}

unsigned int funH(const unsigned int x, const unsigned int y, const unsigned int z) {
    return x ^ y ^ z;
}

unsigned int funI(const unsigned int x, const unsigned int y, const unsigned int z) {
    return y ^ (x | (~z));
}

unsigned int leftRotate(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

#define MAX_WORD_SIZE 10
#define MAX_WORKING_BUFFER_SIZE MAX_WORD_SIZE + 128

void
calculateHashSum(unsigned char *digest, char *words, int workingBufferLength, int lenght) {

    unsigned char workingBuffer[MAX_WORKING_BUFFER_SIZE];
    //init working buffer
    workingBuffer[lenght] = 0b10000000;
    memset(workingBuffer + lenght + 1, 0, workingBufferLength - lenght - 1 - 8);
    reinterpret_cast<unsigned long *>(workingBuffer)[workingBufferLength / 8 - 1] = 8 * lenght;

    workingBuffer[0] = 'd';
    workingBuffer[1] = 'u';

    int combinations = 1;
    for (int i = 0; i < lenght - 2; i++)
        combinations *= 256;

    unsigned int numberOfChunks = workingBufferLength / 64;

    for (long j = 0; j < combinations; j++) {

        memcpy(workingBuffer + 2, &j, (lenght - 2) * sizeof(unsigned char));

        block mdBuffer = DEFAULT_DIGEST_BUFFER;

        for (unsigned long i = 0; i < numberOfChunks; i++) {
//            memcpy(X, workingBuffer + i * 16 * sizeof(unsigned int), 16 * sizeof(unsigned int));
            unsigned int *X = reinterpret_cast<unsigned int *>(workingBuffer + i * 16 * sizeof(unsigned int));
            uint32_t a = mdBuffer.a;
            uint32_t b = mdBuffer.b;
            uint32_t c = mdBuffer.c;
            uint32_t d = mdBuffer.d;

            a = b + leftRotate((a + funF(b, c, d) + X[0] + 0xd76aa478), 7);
            d = a + leftRotate((d + funF(a, b, c) + X[1] + 0xe8c7b756), 12);
            c = d + leftRotate((c + funF(d, a, b) + X[2] + 0x242070db), 17);
            b = c + leftRotate((b + funF(c, d, a) + X[3] + 0xc1bdceee), 22);
            a = b + leftRotate((a + funF(b, c, d) + X[4] + 0xf57c0faf), 7);
            d = a + leftRotate((d + funF(a, b, c) + X[5] + 0x4787c62a), 12);
            c = d + leftRotate((c + funF(d, a, b) + X[6] + 0xa8304613), 17);
            b = c + leftRotate((b + funF(c, d, a) + X[7] + 0xfd469501), 22);
            a = b + leftRotate((a + funF(b, c, d) + X[8] + 0x698098d8), 7);
            d = a + leftRotate((d + funF(a, b, c) + X[9] + 0x8b44f7af), 12);
            c = d + leftRotate((c + funF(d, a, b) + X[10] + 0xffff5bb1), 17);
            b = c + leftRotate((b + funF(c, d, a) + X[11] + 0x895cd7be), 22);
            a = b + leftRotate((a + funF(b, c, d) + X[12] + 0x6b901122), 7);
            d = a + leftRotate((d + funF(a, b, c) + X[13] + 0xfd987193), 12);
            c = d + leftRotate((c + funF(d, a, b) + X[14] + 0xa679438e), 17);
            b = c + leftRotate((b + funF(c, d, a) + X[15] + 0x49b40821), 22);

            a = b + leftRotate((a + funG(b, c, d) + X[1] + 0xf61e2562), 5);
            d = a + leftRotate((d + funG(a, b, c) + X[6] + 0xc040b340), 9);
            c = d + leftRotate((c + funG(d, a, b) + X[11] + 0x265e5a51), 14);
            b = c + leftRotate((b + funG(c, d, a) + X[0] + 0xe9b6c7aa), 20);
            a = b + leftRotate((a + funG(b, c, d) + X[5] + 0xd62f105d), 5);
            d = a + leftRotate((d + funG(a, b, c) + X[10] + 0x02441453), 9);
            c = d + leftRotate((c + funG(d, a, b) + X[15] + 0xd8a1e681), 14);
            b = c + leftRotate((b + funG(c, d, a) + X[4] + 0xe7d3fbc8), 20);
            a = b + leftRotate((a + funG(b, c, d) + X[9] + 0x21e1cde6), 5);
            d = a + leftRotate((d + funG(a, b, c) + X[14] + 0xc33707d6), 9);
            c = d + leftRotate((c + funG(d, a, b) + X[3] + 0xf4d50d87), 14);
            b = c + leftRotate((b + funG(c, d, a) + X[8] + 0x455a14ed), 20);
            a = b + leftRotate((a + funG(b, c, d) + X[13] + 0xa9e3e905), 5);
            d = a + leftRotate((d + funG(a, b, c) + X[2] + 0xfcefa3f8), 9);
            c = d + leftRotate((c + funG(d, a, b) + X[7] + 0x676f02d9), 14);
            b = c + leftRotate((b + funG(c, d, a) + X[12] + 0x8d2a4c8a), 20);

            a = b + leftRotate((a + funH(b, c, d) + X[5] + 0xfffa3942), 4);
            d = a + leftRotate((d + funH(a, b, c) + X[8] + 0x8771f681), 11);
            c = d + leftRotate((c + funH(d, a, b) + X[11] + 0x6d9d6122), 16);
            b = c + leftRotate((b + funH(c, d, a) + X[14] + 0xfde5380c), 23);
            a = b + leftRotate((a + funH(b, c, d) + X[1] + 0xa4beea44), 4);
            d = a + leftRotate((d + funH(a, b, c) + X[4] + 0x4bdecfa9), 11);
            c = d + leftRotate((c + funH(d, a, b) + X[7] + 0xf6bb4b60), 16);
            b = c + leftRotate((b + funH(c, d, a) + X[10] + 0xbebfbc70), 23);
            a = b + leftRotate((a + funH(b, c, d) + X[13] + 0x289b7ec6), 4);
            d = a + leftRotate((d + funH(a, b, c) + X[0] + 0xeaa127fa), 11);
            c = d + leftRotate((c + funH(d, a, b) + X[3] + 0xd4ef3085), 16);
            b = c + leftRotate((b + funH(c, d, a) + X[6] + 0x04881d05), 23);
            a = b + leftRotate((a + funH(b, c, d) + X[9] + 0xd9d4d039), 4);
            d = a + leftRotate((d + funH(a, b, c) + X[12] + 0xe6db99e5), 11);
            c = d + leftRotate((c + funH(d, a, b) + X[15] + 0x1fa27cf8), 16);
            b = c + leftRotate((b + funH(c, d, a) + X[2] + 0xc4ac5665), 23);

            a = b + leftRotate((a + funI(b, c, d) + X[0] + 0xf4292244), 6);
            d = a + leftRotate((d + funI(a, b, c) + X[7] + 0x432aff97), 10);
            c = d + leftRotate((c + funI(d, a, b) + X[14] + 0xab9423a7), 15);
            b = c + leftRotate((b + funI(c, d, a) + X[5] + 0xfc93a039), 21);
            a = b + leftRotate((a + funI(b, c, d) + X[12] + 0x655b59c3), 6);
            d = a + leftRotate((d + funI(a, b, c) + X[3] + 0x8f0ccc92), 10);
            c = d + leftRotate((c + funI(d, a, b) + X[10] + 0xffeff47d), 15);
            b = c + leftRotate((b + funI(c, d, a) + X[1] + 0x85845dd1), 21);
            a = b + leftRotate((a + funI(b, c, d) + X[8] + 0x6fa87e4f), 6);
            d = a + leftRotate((d + funI(a, b, c) + X[15] + 0xfe2ce6e0), 10);
            c = d + leftRotate((c + funI(d, a, b) + X[6] + 0xa3014314), 15);
            b = c + leftRotate((b + funI(c, d, a) + X[13] + 0x4e0811a1), 21);
            a = b + leftRotate((a + funI(b, c, d) + X[4] + 0xf7537e82), 6);
            d = a + leftRotate((d + funI(a, b, c) + X[11] + 0xbd3af235), 10);
            c = d + leftRotate((c + funI(d, a, b) + X[2] + 0x2ad7d2bb), 15);
            b = c + leftRotate((b + funI(c, d, a) + X[9] + 0xeb86d391), 21);

            mdBuffer.a += a;
            mdBuffer.b += b;
            mdBuffer.c += c;
            mdBuffer.d += d;
        }

        if (mdBuffer.a == reinterpret_cast<uint32_t *>(digest)[0] &&
            mdBuffer.b == reinterpret_cast<uint32_t *>(digest)[1] &&
            mdBuffer.c == reinterpret_cast<uint32_t *>(digest)[2] &&
            mdBuffer.d == reinterpret_cast<uint32_t *>(digest)[3])
            memcpy(words, &workingBuffer, lenght);
    }
}
