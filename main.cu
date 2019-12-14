#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include "cuda_clion_hack.hpp"
#include "include/WordsGenerator.h"
#include "include/HexParser.h"
#include "hashing_algorithms/include/HashingArgorithms.h"

const std::string defaultWordsListFileName = "words";

void generateWords(const unsigned int n, const unsigned int length);

void generateDigests(IGenerator *generator);

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
                if (!strcmp(argv[i + 1], "md5_ssl")) {
                    MD5sslDigestGenerator *md5SslDigestGenerator = new MD5sslDigestGenerator();
                    generateDigests(md5SslDigestGenerator);
                    delete md5SslDigestGenerator;
                }
//                else if (!strcmp(argv[i + 1], "md5_cuda"))
//                    generateDigests(MD5cudaDigestGenerator());
            } else std::cout << "too few arguments for -g parameters, correct format -g n length" << std::endl;
        }

    }
}

void generateWords(const unsigned int n, const unsigned int length) {
    std::cout << "start generating " << n << " words list with length set on " << length << std::endl;
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(length, n);
    std::vector<std::string *> *words = wordsGenerator.getWordsBuffer();

    std::ofstream outputFile(defaultWordsListFileName);

    std::cout << "save words list in file: " << defaultWordsListFileName << std::endl;

    outputFile << n << '\t' << length << std::endl;

    for (std::string *word: *words) {
        outputFile << *word << std::endl;
        delete word;
    }

    delete words;

    std::cout << "generate and save words complete" << std::endl;
    outputFile.close();
}

void generateDigests(IGenerator *generator) {
    std::ifstream inputFile(defaultWordsListFileName);
    unsigned int n;
    unsigned int length;

    inputFile >> n;
    inputFile >> length;

    std::cout << "start loading words from file: " << defaultWordsListFileName << std::endl;

    char *buffer = new char[length + 1];
    inputFile.getline(buffer, length + 1);
    char **words = new char *[n];

    for (unsigned int i = 0; i < n; i++) {
        words[i] = new char[length];
        inputFile.getline(buffer, length + 1);
        memcpy(words[i], buffer, length);
    }
    delete buffer;
    inputFile.close();

    std::cout << "loading words complete " << n << " words was loaded" << std::endl;
    std::cout << "initialize generator" << std::endl;
    generator->setLength(length);
    generator->setN(n);
    generator->setWords(words);

    std::cout << "start generating digests" << std::endl;
    generator->generate();
    std::cout << "generation complete" << std::endl;
    std::cout << "printing results" << std::endl;
    unsigned char **digits = generator->getDigits();
    unsigned int md5DigestLength = generator->getDigestLength();
    HexParser md5Parser(md5DigestLength);

    for (unsigned int i = 0; i < n; i++) {
        char *word = words[i];
        std::cout << std::string(word, word + length) << '\t' << md5Parser(digits[i]) << std::endl;
    }

    for (unsigned int i = 0; i < n; i++) {
        delete[] words[i];
        delete[] digits[i];
    }
    delete[]words;
    delete[] digits;

    std::cout << "cleaning memory complete" << std::endl;
}


