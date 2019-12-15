//
// Created by grzegorz on 09.11.2019.
//

#ifndef INYNIERKA_MD5_SSL_H
#define INYNIERKA_MD5_SSL_H

#include "IHashingAlgorithm.cuh"


class MD5_ssl : public IHashingAlgorithm {

    unsigned int defaultWordLength = 0;

public:
    void setDefaultWordLength(unsigned int) override;

    unsigned char *calculateHashSum(const char *word) override;

    unsigned int getDigestLength() override;
};


#endif //INYNIERKA_MD5_SSL_H
