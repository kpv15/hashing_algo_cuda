//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include "IHashingAlgorithm.cuh"

class MD5_cuda : public IHashingAlgorithm {
    std::string calculateHashSum(std::string word) override;

    unsigned char *calculateHashSum(const char *word) override;

};


#endif //INYNIERKA_MD5_CUH
