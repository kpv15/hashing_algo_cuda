//
// Created by grzegorz on 19.01.2020.
//

#include "include/SHA1cpuDigestGenerator.h"

unsigned int SHA1cpuDigestGenerator::getDigestLength() {
    return 20;
}

std::string SHA1cpuDigestGenerator::getAlgorithmName() {
    return "sha1_cpu";
}