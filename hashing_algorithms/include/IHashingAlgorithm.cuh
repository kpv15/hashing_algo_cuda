//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_IHASHINGALGORITHM_CUH
#define INYNIERKA_IHASHINGALGORITHM_CUH


#include <string>

class IHashingAlgorithm {
public:
    virtual void setDefaultWordLength(unsigned int) = 0;

    virtual std::string calculateHashSum(std::string word) = 0;

    virtual char *calculateHashSum(const char *word) = 0;
};


#endif //INYNIERKA_IHASHINGALGORITHM_CUH
