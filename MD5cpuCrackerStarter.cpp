//
// Created by grzegorz on 12.01.2020.
//

#include <iostream>
#include <cstring>
#include <chrono>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/crackers/include/MD5_cpu_cracker.h"

unsigned int calculateWorkingBufferLength(unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return wordLength + toAdd + 8;
}

int crack(int min_length, int max_length, unsigned char *digest);

inline unsigned char hexToInt(unsigned char a, unsigned char b) {
    a = a - '0' < 10 ? a - '0' : a - 'a' + 10;
    b = b - '0' < 10 ? b - '0' : b - 'a' + 10;
    return (a * 16) + b;
}

int main(int argc, char **argv) {

    char digest_hex[DIGEST_LENGTH * 2 + 1];
    unsigned char digest[DIGEST_LENGTH];
    int min = 0;
    int max = 0;
    if (argc >= 4) {
        min = atoi(argv[1]);
        max = atoi(argv[2]);
        strcpy(reinterpret_cast<char *>(&digest_hex), argv[3]);
    }

    for (int i = 0; i < DIGEST_LENGTH; i++) {
        digest[i] = hexToInt(digest_hex[2 * i], digest_hex[2 * i + 1]);
    }

    crack(min, max, digest);

}

int crack(int min_length, int max_length, unsigned char *digest) {

    min_length = min_length >= 2 ? min_length : 2;
    const char NOT_FOUND[] = "-";

    char *word = new char[max_length + 1];

    for (int length = min_length; length <= max_length; length++) {

        memcpy(word, NOT_FOUND, sizeof(char) * (strlen(NOT_FOUND) + 1));

        int workingBufferLength = calculateWorkingBufferLength(length);

        std::cout << "checking word with length: " << length << std::endl;
        auto startKernel = std::chrono::high_resolution_clock::now();

//        auto startKernel = std::chrono::high_resolution_clock::now();

        calculateHashSum(digest, word, workingBufferLength, length);

        auto stopKernel = std::chrono::high_resolution_clock::now();

        word[length] = '\0';

        auto durationKernel = std::chrono::duration_cast<std::chrono::microseconds>(stopKernel - startKernel);

        std::cout << word << "\tin: " << durationKernel.count() << std::endl;

    }

    delete[]word;

    return 0;
}