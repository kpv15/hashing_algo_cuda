#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include "cuda_clion_hack.hpp"
#include "include/WordsGenerator.h"
#include "include/HexParser.h"
#include "hashing_algorithms/include/HashingArgorithms.h"

void generateWords(const unsigned int n, const unsigned int length);

int generateDigests(const IHashingAlgorithm &hashingAlgorithm);

int main(int argc, char **argv) {
    for (unsigned int i = 1; i < argc; i++) {

        if (!strcmp(argv[i], "-g")) {
            if (argc >= i + 2) {
                unsigned int n = atoi(argv[i + 1]);
                unsigned int length = atoi(argv[i + 2]);
                generateWords(n, length);
            } else std::cout << "too few arguments for -g parameters, correct format -g n length" << std::endl;
        }

        if (!strcmp(argv[i], "-d")) {
            if (argc >= i + 1) {
                if (!strcmp(argv[i + 1], "md5_ssl"))
                    generateDigests(MD5_ssl());
                else if (!strcmp(argv[i + 1], "md5_cuda"))
                    generateDigests(MD5_cuda());
            } else std::cout << "too few arguments for -g parameters, correct format -g n length" << std::endl;
        }

    }
}

void generateWords(const unsigned int n, const unsigned int length) {
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(length, n);
    std::vector<std::string *> *words = wordsGenerator.getWordsBuffer();

    std::ofstream myFile("words");

    myFile << words->size() << std::endl;

    for (std::string *word: *words) {
        myFile << *word << std::endl;
        delete word;
    }

    delete words;
}

int generateDigests(const IHashingAlgorithm &hashingAlgorithm) {
//    std::ofstream myFile("words");
//
//    char **words = wordsGenerator.getWordsBufferAsCharArray();
//
//    MD5sslDigestGenerator md5sslDigitsGenerator(words, n, wordLength);
//
//    md5sslDigitsGenerator.generate();
//
//    unsigned char **digits = md5sslDigitsGenerator.getDigits();
//    unsigned int md5DigestLength = md5sslDigitsGenerator.getDigestLength();
//    HexParser md5Parser(md5DigestLength);
//
//    // Create and open a text file
//    std::ofstream MyFile("test_data.txt");
//
//    for (unsigned int i = 0; i < n; i++) {
//        char *word = words[i];
//        std::cout << std::string(word, word + wordLength) << '\t' << md5Parser(digits[i]) << std::endl;
//        MyFile << std::string(word, word + wordLength) << '\t' << md5Parser(digits[i]) << std::endl;
//        delete word;
//        delete digits[i];
//    }
//    delete[] words;
//    delete[] digits;
//
//    MyFile.close();

    return 0;
}


