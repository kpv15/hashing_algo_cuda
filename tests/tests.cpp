//
// Created by grzegorz on 15.12.2019.
//

#include <cstring>
#include <iostream>
#include "../hashing_algorithms/include/MD5_cuda.cuh"
#include "../utils/include/HexParser.h"

bool md5_cuda();

int main() {
    md5_cuda();
}

bool md5_cuda() {
    const std::string TEST_NAME = "md5_cuda";
    unsigned const long WORD_LENGTH = 10;
    char word[WORD_LENGTH];
    unsigned char correctResult[16];
    memcpy(word, "e9ElM4uJMJ", WORD_LENGTH);
    memcpy(correctResult, "Ȝ���xCH[9燠\u00012\u000F�av", 16);

    MD5_cuda md5Cuda;
    md5Cuda.setDefaultWordLength(WORD_LENGTH);
    unsigned char *digest = md5Cuda.calculateHashSum(word);

    for (int i = 0; i < 16; i++)
        if (digest[i] != correctResult[i]) {
            HexParser hexParser(md5Cuda.getDigestLength());
            std::cout << TEST_NAME << " test failed\t[ " << word << " ]\t" << hexParser(digest) << " != "
                      << hexParser(correctResult) << std::endl;
            return false;
        }
    std::cout << TEST_NAME << " test success" << std::endl;

    delete digest;

    return true;
}
