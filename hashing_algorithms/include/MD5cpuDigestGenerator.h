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
    void generate() override {
        md5Cpu.setDefaultWordLength(length_to_gen);
        initDigest();
        for (unsigned int i = 0; i < n_to_gen; i++)
            md5Cpu.calculateHashSum(&digest[i], words[i]);
        n = n_to_gen;
        length = length_to_gen;
    }
    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CPUDIGESTGENERATOR_H
