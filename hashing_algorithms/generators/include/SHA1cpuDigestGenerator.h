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
    void generate() override {
        sha1cpu.setDefaultWordLength(length_to_gen);
        initDigest();
        for (unsigned int i = 0; i < n_to_gen; i++)
            sha1cpu.calculateHashSum(&digest[i], words[i]);
        n = n_to_gen;
        length = length_to_gen;
    }
    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_SHA1CPUDIGESTGENERATOR_H
