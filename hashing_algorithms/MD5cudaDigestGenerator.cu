//
// Created by grzegorz on 15.12.2019.
//


#include "include/MD5cudaDigestGenerator.cuh"

std::string MD5cudaDigestGenerator::getAlgorithmName() {
    return "md5_cuda";
}

void MD5cudaDigestGenerator::setN(unsigned int n) {
    this->n=n;
}

void MD5cudaDigestGenerator::setLength(unsigned int length) {
    this->length = length;
}

unsigned char **MD5cudaDigestGenerator::getDigits() {
    unsigned char **toReturn = digest;
    digest = nullptr;
    return toReturn;
}

void MD5cudaDigestGenerator::setWords(char **words) {
    this->words = words;
}

unsigned int MD5cudaDigestGenerator::getDigestLength() {
    return md5Cuda.getDigestLength();
}
