//
// Created by grzegorz on 23.11.2019.
//

#include "include/MD5sslDigestGenerator.h"
#include "include/MD5_ssl.h"

void MD5sslDigestGenerator::setWords(char **words) {
    MD5sslDigestGenerator::words = words;
}

unsigned char **MD5sslDigestGenerator::getDigits() {
    unsigned char **toReturn = digest;
    digest = nullptr;
    return toReturn;
}

void MD5sslDigestGenerator::setN(const unsigned int n) {
    MD5sslDigestGenerator::n_to_gen = n;
}


void MD5sslDigestGenerator::setLength(const unsigned int length) {
    MD5sslDigestGenerator::length_to_gen = length;
}

void MD5sslDigestGenerator::generate() {
    md5Ssl.setDefaultWordLength(length_to_gen);
    initDigest();

    for (unsigned int i = 0; i < n_to_gen; i++)
        digest[i] = md5Ssl.calculateHashSum(words[i]);

    length = length_to_gen;
    n = n_to_gen;
}

void MD5sslDigestGenerator::initDigest() {
    if (digest != nullptr)
        for (unsigned int i = 0; i < n; i++)
            delete digest[i];
    delete[] digest;

    digest = new unsigned char *[n_to_gen];

}

unsigned int MD5sslDigestGenerator::getDigestLength() {
    return md5Ssl.getDigestLength();
}

std::string MD5sslDigestGenerator::getAlgorithmName() {
    return "md5_ssl";
}
