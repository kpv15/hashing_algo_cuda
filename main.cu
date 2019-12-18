#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include "cuda_clion_hack.hpp"
#include "utils/include/WordsGenerator.h"
#include "utils/include/HexParser.h"
#include "hashing_algorithms/include/HashingAlgorithms.h"
#include "utils/include/ResultComparator.h"

const std::string DEFAULT_WORD_LIST_OUTPUT_FILE_NAME = "words";

void generateWords(const unsigned int n, const unsigned int length);

void generateDigests(IGenerator *generator);

void compareResults(std::vector<std::string> filesNames);

int main(int argc, char **argv) {

    int length = 0;
    int n = 0;
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
                    fileList.push_back(md5SslDigestGenerator->getAlgorithmName() + ".hex");
                    delete md5SslDigestGenerator;
                } else if (!strcmp(argv[i + 1], "md5_cuda")) {
                    MD5cudaDigestGenerator *md5cudaDigestGenerator = new MD5cudaDigestGenerator();
                    generateDigests(md5cudaDigestGenerator);
                    fileList.push_back(md5cudaDigestGenerator->getAlgorithmName() + ".hex");
                    delete md5cudaDigestGenerator;
                }
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

    std::ofstream outputFile(DEFAULT_WORD_LIST_OUTPUT_FILE_NAME);

    std::cout << "save words list in file: " << DEFAULT_WORD_LIST_OUTPUT_FILE_NAME << std::endl;

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
    std::ifstream inputFile(DEFAULT_WORD_LIST_OUTPUT_FILE_NAME);
    unsigned int n = 0;
    unsigned int length = 0;

    inputFile >> n;
    inputFile >> length;
    std::cout << "####################" << std::endl;
    std::cout << "start loading words from file: " << DEFAULT_WORD_LIST_OUTPUT_FILE_NAME << std::endl;

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
    generator->setWords(words, n, length);

    std::cout << "start generating digests" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    generator->generate();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "generation complete in: " << duration.count() << " millisecondsl" << std::endl;


    std::string algorithmName = generator->getAlgorithmName();
    std::ofstream outputDigestHex(algorithmName + ".hex");
    std::ofstream outputDigest(algorithmName + ".txt");

    std::cout << "open output file: " << algorithmName << std::endl;
    std::cout << "printing results" << std::endl;
    unsigned char **digits = generator->getDigits();
    unsigned int digestLength = generator->getDigestLength();
    HexParser hexParser(digestLength);

    if (digits != nullptr) {
        for (unsigned int i = 0; i < n; i++) {
            char *word = words[i];
            std::cout << std::string(word, word + length) << "\t" << hexParser(digits[i]) << std::endl;
            outputDigest << hexParser(digits[i]) << std::endl;
            outputDigestHex.write((char *) (digits[i]), digestLength);
        }

        std::cout << "output file closed, cleaning memory" << std::endl;
        for (unsigned int i = 0; i < n; i++) {
            delete[] words[i];
            delete[] digits[i];
        }
        delete[] words;
        delete[] digits;

        outputDigest.close();
        outputDigestHex.close();

        std::cout << "cleaning memory complete" << std::endl;
    }
}

void compareResults(std::vector<std::string> filesNames) {
    ResultComparator resultComparator(filesNames);
    resultComparator.compare();
}
