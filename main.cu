#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include "cuda_clion_hack.hpp"
#include "include/WordsGenerator.h"
#include "include/HexParser.h"
#include "hashing_algorithms/include/HashingAlgorithms.h"
#include "hashing_algorithms/ResultComparator.h"

const std::string defaultWordsListOutputFileName = "words";

void generateWords(const unsigned int n, const unsigned int length);

void generateDigests(IGenerator *generator);

void compareResults(std::vector<std::string> filesNames);

int main(int argc, char **argv) {

    unsigned int length = 0;
    unsigned int n = 0;
    std::vector<std::string> fileList;

    for (unsigned int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n")) {
            if (argc >= i + 1)
                n = atoi(argv[i + 1]);
            else
                std::cout << "after -n should be given number of records" << std::endl;
        } else if (!strcmp(argv[i], "-l")) {
            if (argc >= i + 1)
                length = atoi(argv[i + 1]);
            else
                std::cout << "after -l should be given length of word" << std::endl;
        }
    }

    for (unsigned int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-g")) {
            if (length <= 0 || n <= 0) {
                std::cout << "provide number of records and length of word" << std::endl;
                return 1;
            }
            generateWords(n, length);
        }
    }

    for (unsigned int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d")) {
            if (argc >= i + 1) {
                if (!strcmp(argv[i + 1], "md5_ssl")) {
                    MD5sslDigestGenerator *md5SslDigestGenerator = new MD5sslDigestGenerator();
                    generateDigests(md5SslDigestGenerator);
                    fileList.push_back(md5SslDigestGenerator->getAlgorithmName());
                    delete md5SslDigestGenerator;
                }
//                else if (!strcmp(argv[i + 1], "md5_cuda"))
//                    generateDigests(MD5cudaDigestGenerator());
            } else std::cout << "too few arguments for -g parameters, correct format -g n length" << std::endl;
        }
    }

    for (unsigned int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-c")) {
            compareResults(fileList);
        }
}

void generateWords(const unsigned int n, const unsigned int length) {
    std::cout << "start generating " << n << " words list with length set on " << length << std::endl;
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(length, n);
    std::vector<std::string *> *words = wordsGenerator.getWordsBuffer();

    std::ofstream outputFile(defaultWordsListOutputFileName);

    std::cout << "save words list in file: " << defaultWordsListOutputFileName << std::endl;

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
    std::ifstream inputFile(defaultWordsListOutputFileName);
    unsigned int n;
    unsigned int length;

    inputFile >> n;
    inputFile >> length;

    std::cout << "start loading words from file: " << defaultWordsListOutputFileName << std::endl;

    char *buffer = new char[length + 1];
    inputFile.getline(buffer, length + 1);
    char **words = new char *[n];

    for (unsigned int i = 0; i < n; i++) {
        words[i] = new char[length];
        inputFile.getline(buffer, length + 1);
        memcpy(words[i], buffer, length);
    }
    delete[] buffer;
    inputFile.close();

    std::cout << "loading words complete " << n << " words was loaded, input file closed" << std::endl;
    std::cout << "initialize generator" << std::endl;
    generator->setLength(length);
    generator->setN(n);
    generator->setWords(words);

    std::cout << "start generating digests" << std::endl;
    generator->generate();
    std::cout << "generation complete" << std::endl;

    std::string algorithmName = generator->getAlgorithmName();
    std::ofstream outputDigest(algorithmName);

    std::cout << "open output file: " << algorithmName << std::endl;
    std::cout << "printing results" << std::endl;
    unsigned char **digits = generator->getDigits();
    unsigned int digestLength = generator->getDigestLength();
    HexParser hexParser(digestLength);//todo resolve parser type problem

    for (unsigned int i = 0; i < n; i++) {
        char *word = words[i];
        std::cout << std::string(word, word + length) << '\t' << hexParser(digits[i]) << std::endl;
        outputDigest << hexParser(digits[i]) << std::endl;
    }

    std::cout << "output file closed, cleaning memory" << std::endl;
    for (unsigned int i = 0; i < n; i++) {
        delete[] words[i];
        delete[] digits[i];
    }
    delete[] words;
    delete[] digits;

    std::cout << "cleaning memory complete" << std::endl;
}

void compareResults(std::vector<std::string> filesNames) {
    ResultComparator resultComparator(filesNames);
    resultComparator.compare();
}
