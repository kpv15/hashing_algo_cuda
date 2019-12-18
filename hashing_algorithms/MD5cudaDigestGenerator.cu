//
// Created by grzegorz on 15.12.2019.
//


#include "include/MD5cudaDigestGenerator.cuh"

std::string MD5cudaDigestGenerator::getAlgorithmName() {
    return "md5_cuda";
}

unsigned int MD5cudaDigestGenerator::getDigestLength() {
    return 16;
}
