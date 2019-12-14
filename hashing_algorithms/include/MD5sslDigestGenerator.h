//
// Created by grzegorz on 23.11.2019.
//

#ifndef INYNIERKA_MD5SSLDIGESTGENERATOR_H
#define INYNIERKA_MD5SSLDIGESTGENERATOR_H


#include "MD5_ssl.h"
#include "IGenerator.h"

class MD5sslDigestGenerator : public IGenerator {
    char **words = nullptr;
    unsigned char **digest = nullptr;
    unsigned int n = 0;
    unsigned int length = 0;
    unsigned int n_to_gen = 0;
    unsigned int length_to_gen = 0;
    MD5_ssl md5Ssl;

    void initDigest();

public:
    void setN(unsigned int n) override;

    void setLength(unsigned int length) override;

    unsigned char **getDigits() override;

    void setWords(char **words) override;

    void generate() override;

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5SSLDIGESTGENERATOR_H
