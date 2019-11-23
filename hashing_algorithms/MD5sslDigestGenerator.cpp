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

void MD5sslDigestGenerator::setN(unsigned int n) {
    MD5sslDigestGenerator::n_to_gen = n;
}


void MD5sslDigestGenerator::setLength(unsigned int length) {
    MD5sslDigestGenerator::length_to_gen = length;
}

MD5sslDigestGenerator::MD5sslDigestGenerator(char **words, unsigned int n, unsigned int length) : words(words),
                                                                                                  n_to_gen(n),
                                                                                                  length_to_gen(
                                                                                                          length) {}

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
