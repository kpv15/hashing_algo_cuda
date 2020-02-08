//
// Created by grzegorz on 19.12.2019.
//

#ifndef INYNIERKA_MD5CPUDIGESTGENERATOR_H
#define INYNIERKA_MD5CPUDIGESTGENERATOR_H


#include "IGenerator.h"
#include "MD5_cpu.h"

class MD5cpuDigestGenerator: public IGenerator {
    MD5_cpu md5Cpu;
public:
    void generate() override;
    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CPUDIGESTGENERATOR_H
