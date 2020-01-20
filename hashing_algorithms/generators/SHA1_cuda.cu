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

    __global__ void calculateHashSum(unsigned char *digest, char *word, unsigned long int workingBufferLength,
                                     unsigned long int wordLength, unsigned long int n){
        //todo implement
    }
}