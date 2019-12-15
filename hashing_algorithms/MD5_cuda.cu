//
// Created by grzegorz on 06.11.2019.
//

#include "include/MD5_cuda.cuh"

void MD5_cuda::setDefaultWordLength(unsigned int i) {
    this->defaultWordLength = i;
}

unsigned int MD5_cuda::getDigestLength() {
    return DIGEST_LENGTH;
}

unsigned int MD5_cuda::calculateWorkingBufferLength(const unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return wordLength + toAdd;
}
