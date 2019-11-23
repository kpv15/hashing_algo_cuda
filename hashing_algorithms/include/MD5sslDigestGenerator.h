//
// Created by grzegorz on 23.11.2019.
//

#ifndef INYNIERKA_MD5SSLDIGESTGENERATOR_H
#define INYNIERKA_MD5SSLDIGESTGENERATOR_H


#include "MD5_ssl.h"

class MD5sslDigestGenerator {
    char **words = nullptr;
    unsigned char **digest = nullptr;
    unsigned int n;
    unsigned int length;
    unsigned int n_to_gen = 0;
    unsigned int length_to_gen = 0;
    MD5_ssl md5Ssl;

    void initDigest();

public:
    void setN(unsigned int n);

    void setLength(unsigned int length);

    unsigned char **getDigits();

    void setWords(char **words);

    MD5sslDigestGenerator(char **words, unsigned int n, unsigned int length);

    void generate();

    unsigned int getDigestLength();
};


#endif //INYNIERKA_MD5SSLDIGESTGENERATOR_H
