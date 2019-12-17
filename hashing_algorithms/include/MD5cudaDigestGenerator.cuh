//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
#define INYNIERKA_MD5CUDADIGESTGENERATOR_CUH


#include "MD5_cpu.h"
#include "IGenerator.h"

class MD5cudaDigestGenerator : public IGenerator {

    MD5_cpu md5Cuda;

public:
    void generate() override {
        md5Cuda.setDefaultWordLength(length_to_gen);
        initDigest();
        for (unsigned int i = 0; i < n_to_gen; i++)
            digest[i] = md5Cuda.calculateHashSum(words[i]);
        n = n_to_gen;
        length = length_to_gen;
    }

    unsigned int getDigestLength() override;

    std::string getAlgorithmName() override;
};


#endif //INYNIERKA_MD5CUDADIGESTGENERATOR_CUH
