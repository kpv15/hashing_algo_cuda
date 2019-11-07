//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_MD5_CUH
#define INYNIERKA_MD5_CUH

#include "IHashingAlgorithm.cuh"

class MD5 : private IHashingAlgorithm {
    std::string calculateHashSum(std::string word) override;
};


#endif //INYNIERKA_MD5_CUH
