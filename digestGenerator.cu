#include <iostream>
#include <vector>
#include "cuda_clion_hack.hpp"
#include "hashing_algorithms/include/IHashingAlgorithm.cuh"
#include "hashing_algorithms/include/MD5_ssl.h"
#include "include/WordsGenerator.h"
#include "include/HexParser.h"
#include "hashing_algorithms/include/MD5sslDigestGenerator.h"


int main(void) {
    const unsigned int wordLength = 16;
    const unsigned int n = 2;
    WordsGenerator wordsGenerator;
    wordsGenerator.generate(wordLength, n);
    char **words = wordsGenerator.getWordsBufferAsCharArray();

    MD5sslDigestGenerator md5sslDigitsGenerator(words, n, wordLength);

    md5sslDigitsGenerator.generate();

    unsigned char **digits = md5sslDigitsGenerator.getDigits();
    unsigned int md5DigestLength = md5sslDigitsGenerator.getDigestLength();
    HexParser md5Parser(md5DigestLength);

    for (unsigned int i = 0; i < n; i++) {
        char *word = words[i];
        std::cout << std::string(word, word + md5DigestLength) << '\t' << md5Parser(digits[i]) << std::endl;
        delete word;
        delete digits[i];
    }
    delete words;
    delete digits;


    return 0;
}