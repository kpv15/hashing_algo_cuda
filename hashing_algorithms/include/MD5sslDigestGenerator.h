//
// Created by grzegorz on 23.11.2019.
//

#ifndef INYNIERKA_MD5SSLDIGESTGENERATOR_H
#define INYNIERKA_MD5SSLDIGESTGENERATOR_H


#include "MD5_ssl.h"
#include "IGenerator.h"

class MD5sslDigestGenerator : public IGenerator {
    MD5_ssl md5Ssl;

public:
    void generate() override;

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5SSLDIGESTGENERATOR_H
