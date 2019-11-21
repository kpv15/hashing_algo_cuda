#include <iostream>
#include <vector>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/IHashingAlgorithm.cuh"
#include "hashing_algorithms/include/MD5_ssl.h"
#include "include/WordsGenerator.h"
#include "include/HexParser.h"


int main(void) {
    const unsigned int wordLenght = 16;
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(wordLenght, 1);
    std::vector<std::string *> *randStrings = wordsGenerator.getWordsBuffer();
    IHashingAlgorithm *algorithm = new MD5_ssl(wordLenght);
    HexParser md5Parser(std::cout, algorithm->getDigestLength());

    for (std::string *word: *randStrings) {
        std::cout << *word << "\t";
        std::string digest = algorithm->calculateHashSum(*word);
        md5Parser.print(digest);
        std::cout << "\n";
    }


    return 0;
}