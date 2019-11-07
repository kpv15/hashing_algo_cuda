#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/IHashingAlgorithm.cuh"
#include "hashing_algorithms/include/MD5.cuh"
#include "include/WordsGenerator.h"

int main(void) {
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(10, 8);
    std::vector<std::string*> *randStrings = wordsGenerator.getWordsBuffer();

    for (std::string *word: *randStrings)
        std::cout << *word << std::endl;

    return 0;
}