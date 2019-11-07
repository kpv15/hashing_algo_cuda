//
// Created by grzegorz on 06.11.2019.
//

#ifndef INYNIERKA_WORDSGENERATOR_H
#define INYNIERKA_WORDSGENERATOR_H

#include <iostream>
#include <vector>
#include <string>


class WordsGenerator {
    std::vector<std::string *> *words_buffer = nullptr;
    unsigned int lenght = 0;

    void freeBuffer();

public:
    void generate(unsigned int lenght, unsigned int n);

    std::vector<std::string *> *getWordsBuffer();

    char **getWordsBufferAsCharArray();

    virtual ~WordsGenerator();
};


#endif //INYNIERKA_WORDSGENERATOR_H
