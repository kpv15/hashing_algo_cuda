//
// Created by grzegorz on 14.12.2019.
//

#ifndef INYNIERKA_IGENERATOR_H
#define INYNIERKA_IGENERATOR_H

class IGenerator {
public:
    virtual void setN(unsigned int n) = 0;

    virtual void setLength(unsigned int length) = 0;

    virtual unsigned char **getDigits() = 0;

    virtual void setWords(char **words) = 0;

    virtual void generate() = 0;

    virtual unsigned int getDigestLength() = 0;
};

#endif //INYNIERKA_IGENERATOR_H
