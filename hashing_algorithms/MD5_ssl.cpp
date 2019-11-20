//
// Created by grzegorz on 09.11.2019.
//

#include "include/MD5_ssl.h"
#include <openssl/md5.h>

std::string MD5_ssl::calculateHashSum(std::string word) {
    char *digest = (calculateHashSum(word.c_str()));

    std::string toReturn(digest, digest + MD5_DIGEST_LENGTH * 2);

    delete[] digest;

    return toReturn;
}

char *MD5_ssl::calculateHashSum(const char *word) {
    unsigned char result[MD5_DIGEST_LENGTH];

    MD5(reinterpret_cast<const unsigned char *>(word), defaultWordLength, result);
    char *digest = new char[MD5_DIGEST_LENGTH * 2];

    for (int i = 0; i < 16; i++) {
        sprintf(&digest[i * 2], "%02x", result[i]);
        printf("%02x", result[i]);
    }
    printf(" ");

    return digest;
}

void MD5_ssl::setDefaultWordLength(unsigned int defaultLength) {
    defaultWordLength = defaultLength;
}
