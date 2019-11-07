//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_IHASHINGALGORITHM_CUH
#define INYNIERKA_IHASHINGALGORITHM_CUH


#include <string>

class IHashingAlgorithm {
    virtual std::string calculateHashSum(std::string word) = 0;
};


#endif //INYNIERKA_IHASHINGALGORITHM_CUH
