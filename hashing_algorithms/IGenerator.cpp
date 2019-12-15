//
// Created by grzegorz on 15.12.2019.
#include "include/IGenerator.h"

//

void IGenerator::initDigest() {
    if (digest != nullptr)
        for (unsigned int i = 0; i < n; i++)
            delete digest[i];
    delete[] digest;

    digest = new unsigned char *[n_to_gen];
}

unsigned char **IGenerator::getDigits() {
    unsigned char **toReturn = digest;
    digest = nullptr;
    return toReturn;
}

void IGenerator::setWords(char **words, unsigned int n, unsigned int length) {
    this->n_to_gen = n;
    this->length_to_gen = length;
    this->words = words;
}
