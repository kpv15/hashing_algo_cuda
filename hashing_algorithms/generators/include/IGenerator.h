//
// Created by grzegorz on 14.12.2019.
//

#ifndef INYNIERKA_IGENERATOR_H
#define INYNIERKA_IGENERATOR_H

#include <string>

class IGenerator {
protected:
    char **words = nullptr;
    unsigned char **digest = nullptr;
    unsigned int n = 0;
    unsigned int length = 0;
    unsigned int n_to_gen = 0;
    unsigned int length_to_gen = 0;

    void initDigest();

public:
    virtual unsigned char **getDigits();

    virtual void setWords(char **words, unsigned int n, unsigned int length);

    virtual void generate() = 0;

    virtual unsigned int getDigestLength() = 0;

    virtual std::string getAlgorithmName() = 0;

    virtual bool needOneDimArray() { return false;}
};

#endif //INYNIERKA_IGENERATOR_H
