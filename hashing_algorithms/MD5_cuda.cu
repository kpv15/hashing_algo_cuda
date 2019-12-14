//
// Created by grzegorz on 06.11.2019.
//

#include "include/MD5_cuda.cuh"

void MD5_cuda::setDefaultWordLength(unsigned int i) {
    this->defaultWordLength = i;
}

unsigned int MD5_cuda::getDigestLength() {
    return digestLength;
}

unsigned int MD5_cuda::calculateWorkingLength(const unsigned int wordLenth) {
    return ((((wordLenth + 8) / 64) + 1) * 64) - 8;
}
