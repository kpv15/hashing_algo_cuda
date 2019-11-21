//
// Created by grzegorz on 06.11.2019.
//

#include <cstring>
#include "include/WordsGenerator.h"

static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

void WordsGenerator::generate(unsigned int lenght, unsigned int n) {
    freeBuffer();
    words_buffer = new std::vector<std::string *>();
    this->lenght = lenght;

    for (unsigned int i = 0; i < n; ++i) {
        auto *word = new std::string();
        for (unsigned int j = 0; j < lenght; ++j)
            *word += alphanum[rand() % (sizeof(alphanum) - 1)];
        words_buffer->emplace_back(word);
    }
}

std::vector<std::string *> *WordsGenerator::getWordsBuffer() {
    std::vector<std::string *> *toReturn = words_buffer;
    words_buffer = nullptr;
    return toReturn;
}

WordsGenerator::~WordsGenerator() {
    freeBuffer();
}

char **WordsGenerator::getWordsBufferAsCharArray() {
    char **toReturn = nullptr;
    if (words_buffer != nullptr) {
        unsigned int n = words_buffer->size();
        toReturn = new char *[n];
        for (unsigned int i = 0; i < n; ++i) {
            toReturn[i] = new char[lenght];
            std::string *word = (*words_buffer)[i];
            strcpy(toReturn[i], word->c_str());
        }
        freeBuffer();
    }
    return toReturn;
}

void WordsGenerator::freeBuffer() {
    if (words_buffer != nullptr) {
        for (std::string *word: *words_buffer)
            delete word;
        delete[] words_buffer;
    }
}
