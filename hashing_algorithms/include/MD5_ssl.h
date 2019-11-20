//
// Created by grzegorz on 09.11.2019.
//

#ifndef INYNIERKA_MD5_SSL_H
#define INYNIERKA_MD5_SSL_H

#include "IHashingAlgorithm.cuh"


class MD5_ssl : public IHashingAlgorithm {

    unsigned int defaultWordLength = 0;
public:
    MD5_ssl(unsigned int defaultWordLength) : defaultWordLength(defaultWordLength) {};

    void setDefaultWordLength(unsigned int) override;

    std::string calculateHashSum(std::string word) override;

    char *calculateHashSum(const char *word) override;

};


#endif //INYNIERKA_MD5_SSL_H
