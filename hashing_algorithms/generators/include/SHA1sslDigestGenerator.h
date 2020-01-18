//
// Created by grzegorz on 19.01.2020.
//

#ifndef INYNIERKA_SHA1SSLDIGESTGENERATOR_H
#define INYNIERKA_SHA1SSLDIGESTGENERATOR_H


#include "SHA1_ssl.h"
#include "IGenerator.h"

class SHA1sslDigestGenerator : public IGenerator{
    SHA1_ssl sha1ssl;

public:
    void generate() override;

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_SHA1SSLDIGESTGENERATOR_H
