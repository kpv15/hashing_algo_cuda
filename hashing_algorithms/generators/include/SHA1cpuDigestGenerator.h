//
// Created by grzegorz on 19.01.2020.
//

#ifndef INYNIERKA_SHA1CPUDIGESTGENERATOR_H
#define INYNIERKA_SHA1CPUDIGESTGENERATOR_H


#include "IGenerator.h"
#include "SHA1_cpu.h"

class SHA1cpuDigestGenerator : public IGenerator{
    SHA1_cpu sha1cpu;
public:
    void generate() override;
    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_SHA1CPUDIGESTGENERATOR_H
