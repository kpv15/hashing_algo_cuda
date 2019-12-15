//
// Created by grzegorz on 23.11.2019.
//

#include "include/MD5sslDigestGenerator.h"
#include "include/MD5_ssl.h"


void MD5sslDigestGenerator::generate() {
    md5Ssl.setDefaultWordLength(length_to_gen);
    initDigest();

    for (unsigned int i = 0; i < n_to_gen; i++)
        digest[i] = md5Ssl.calculateHashSum(words[i]);

    length = length_to_gen;
    n = n_to_gen;
}

unsigned int MD5sslDigestGenerator::getDigestLength() {
    return md5Ssl.getDigestLength();
}

std::string MD5sslDigestGenerator::getAlgorithmName() {
    return "md5_ssl";
}
