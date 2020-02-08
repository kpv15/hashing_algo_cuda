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

void MD5cpuDigestGenerator::generate() {
    md5Cpu.setDefaultWordLength(length_to_gen);
    initDigest();
    for (unsigned int i = 0; i < n_to_gen; i++)
        md5Cpu.calculateHashSum(&digest[i], words[i]);
    n = n_to_gen;
    length = length_to_gen;
}
