//
// Created by grzegorz on 19.01.2020.
//

#ifndef INYNIERKA_SHA1_SSL_H
#define INYNIERKA_SHA1_SSL_H


#include "IHashingAlgorithm.h"

class SHA1_ssl: public IHashingAlgorithm {
    unsigned int defaultWordLength = 0;

public:
    void setDefaultWordLength(unsigned int) override;

    void calculateHashSum(unsigned char **digest,const char *word) override;

    unsigned int getDigestLength() override;
};


#endif //INYNIERKA_SHA1_SSL_H
