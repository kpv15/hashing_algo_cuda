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
    unsigned char correctResult[16] = {
            0xb1, 0xee, 0x53, 0xbe,
            0x2e, 0xc3, 0x1e, 0xa1,
            0xa9, 0xae, 0x8a, 0x17,
            0xb5, 0xf3, 0x4c, 0x36
    };
    memcpy(word, "d0190uJXL3", WORD_LENGTH);

    MD5_cuda md5Cuda;
    md5Cuda.setDefaultWordLength(WORD_LENGTH);
    unsigned char *digest = md5Cuda.calculateHashSum(word);

    for (int i = 0; i < 16; i++)
        if (digest[i] != correctResult[i]) {
            HexParser hexParser(md5Cuda.getDigestLength());
            std::cout << TEST_NAME << " test failed\t[ "
                      << std::string(word, word + WORD_LENGTH) << " ]\t"
                      << hexParser(digest) << " != "
                      << hexParser(correctResult) << std::endl;
            return false;
        }
    std::cout << TEST_NAME << " test success" << std::endl;

    delete digest;

    return true;
}
