#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/IHashingAlgorithm.cuh"
#include "hashing_algorithms/include/MD5_ssl.h"
#include "include/WordsGenerator.h"

int main(void) {
    const unsigned int wordLenght = 16;
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(wordLenght, 4);
    std::vector<std::string *> *randStrings = wordsGenerator.getWordsBuffer();
    IHashingAlgorithm *algorithm = new MD5_ssl(wordLenght);


    for (std::string *word: *randStrings) {
        std::cout << *word << "\t";
        std::cout << std::hex << algorithm->calculateHashSum(*word) << std::endl;
    }


    return 0;
}