//
// Created by grzegorz on 15.01.2020.
//

#include "include/SHA1_cpu.h"

const SHA1_cpu::block SHA1_cpu::DEFAULT_DIGEST_BUFFER = {
        0x67452301,
        0xEFCDAB89,
        0x98BADCFE,
        0x10325476,
        0xC3D2E1F0
};

unsigned int SHA1_cpu::calculateWorkingBufferLength() {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return defaultWordLength + toAdd + 8;
}

SHA1_cpu::~SHA1_cpu() {
    delete[] workingBuffer;
}

void SHA1_cpu::setDefaultWordLength(unsigned int i) {
    this->defaultWordLength = i;
}

unsigned int SHA1_cpu::getDigestLength() {
    return DIGEST_LENGTH;
}

void SHA1_cpu::calculateHashSum(uint8_t **digest, const char *word) {

    createWorkingBuffer(word);
    uint32_t w[80];

    block mdBuffer = DEFAULT_DIGEST_BUFFER;
    block stepBuffer;
    uint32_t temp;

    for (unsigned int chunkNum = 0; chunkNum < numberOfChunks; chunkNum++) {
        memcpy(w, workingBuffer + chunkNum * 16 * sizeof(uint32_t), 16 * sizeof(uint32_t));

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
                temp = leftRotate(stepBuffer.a, 5) + funH( stepBuffer.b, stepBuffer.c, stepBuffer.d) +
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

    mdBuffer.a = __builtin_bswap32(mdBuffer.a);
    mdBuffer.b = __builtin_bswap32(mdBuffer.b);
    mdBuffer.c = __builtin_bswap32(mdBuffer.c);
    mdBuffer.d = __builtin_bswap32(mdBuffer.d);
    mdBuffer.e = __builtin_bswap32(mdBuffer.e);

    *digest = new unsigned char[DIGEST_LENGTH];
    memcpy(*digest, &mdBuffer, DIGEST_LENGTH);
}

uint32_t swap_bits(uint32_t x) {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(&x);
    return (ptr[3] << 0) | (ptr[2] << 8) | (ptr[1] << 16) | (ptr[0] << 24);
}

void SHA1_cpu::createWorkingBuffer(const char *word) {
    unsigned long int calculatedWorkingBufferLength = calculateWorkingBufferLength();
    if (workingBuffer != nullptr && calculatedWorkingBufferLength != workingBufferLength)
        delete[] workingBuffer;
    if (workingBuffer == nullptr) {
        workingBuffer = new unsigned char[calculatedWorkingBufferLength];
        workingBufferLength = calculatedWorkingBufferLength;
        numberOfChunks = workingBufferLength / 64;
    }

    unsigned int i = 0, j;
    uint32_t *word_ptr = (uint32_t *) word;
    uint32_t *workingbuffer_ptr = (uint32_t *) workingBuffer;
    for (i = 0; i < defaultWordLength / 4; i++)
        workingbuffer_ptr[i] = swap_bits(word_ptr[i]);
    i = i * 4;

    while (i < defaultWordLength) {
        j = (i / 4) * 4 + 3 - (i % 4);
        workingBuffer[j] = word[i];
        i++;
    }
    j = (i / 4) * 4 + 3 - (i % 4);
    workingBuffer[j] = 0b10000000;
    i++;
    while (i < workingBufferLength - 2) {
        j = (i / 4) * 4 + 3 - (i % 4);
        workingBuffer[j] = 0b00000000;
        i++;
    }

    uint64_t tmp = defaultWordLength * 8;
    std::memcpy(workingBuffer + workingBufferLength - 4, (uint32_t *) &tmp, sizeof(uint32_t));
    std::memcpy(workingBuffer + workingBufferLength - 8, (uint32_t *) &tmp + 1, sizeof(uint32_t));

}

uint32_t SHA1_cpu::leftRotate(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

uint32_t SHA1_cpu::funI(const uint32_t b, const uint32_t c, const uint32_t d) {
    return b ^ c ^ d;
}

uint32_t SHA1_cpu::funH(const uint32_t b, const uint32_t c, const uint32_t d) {
    return (b & c) | (b & d) | (c & d);
}

uint32_t SHA1_cpu::funG(const uint32_t b, const uint32_t c, const uint32_t d) {
    return b ^ c ^ d;
}

uint32_t SHA1_cpu::funF(const uint32_t b, const uint32_t c, const uint32_t d) {
    return (b & c) | ((~b) & d);
}
