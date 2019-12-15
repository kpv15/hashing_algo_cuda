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

unsigned int MD5_cuda::calculateWorkingBufferLength() {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return defaultWordLength + toAdd + 8;
}
