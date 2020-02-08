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

void SHA1cpuDigestGenerator::generate() {
    sha1cpu.setDefaultWordLength(length_to_gen);
    initDigest();
    for (unsigned int i = 0; i < n_to_gen; i++)
        sha1cpu.calculateHashSum(&digest[i], words[i]);
    n = n_to_gen;
    length = length_to_gen;
}
