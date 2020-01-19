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