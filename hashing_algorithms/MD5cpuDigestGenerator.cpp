//
// Created by grzegorz on 19.12.2019.
//

#include "include/MD5cpuDigestGenerator.h"

unsigned int MD5cpuDigestGenerator::getDigestLength() {
    return 16;
}

std::string MD5cpuDigestGenerator::getAlgorithmName() {
    return "md5_cpu";
}
