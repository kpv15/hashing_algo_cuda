//
// Created by grzegorz on 19.01.2020.
//

#include "include/SHA1sslDigestGenerator.h"
#include "include/SHA1_ssl.h"

void SHA1sslDigestGenerator::generate() {
    sha1ssl.setDefaultWordLength(length_to_gen);
    initDigest();

    for (unsigned int i = 0; i < n_to_gen; i++)
        sha1ssl.calculateHashSum(&digest[i], words[i]);

    length = length_to_gen;
    n = n_to_gen;
}

unsigned int SHA1sslDigestGenerator::getDigestLength() {
    return sha1ssl.getDigestLength();
}

std::string SHA1sslDigestGenerator::getAlgorithmName() {
    return "sha1_ssl";
}