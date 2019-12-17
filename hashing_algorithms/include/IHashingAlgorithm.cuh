//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_IHASHINGALGORITHM_CUH
#define INYNIERKA_IHASHINGALGORITHM_CUH


#include <string>

class IHashingAlgorithm {

public:

    virtual void setDefaultWordLength(unsigned int) = 0;

    virtual void calculateHashSum(unsigned char **digest,const char *word) = 0;

    virtual unsigned int getDigestLength() = 0;

};


#endif //INYNIERKA_IHASHINGALGORITHM_CUH
