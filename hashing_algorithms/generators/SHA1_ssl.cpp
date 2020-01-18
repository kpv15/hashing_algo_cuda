//
// Created by grzegorz on 19.01.2020.
//

#include "include/SHA1_ssl.h"
#include <openssl/sha.h>

void SHA1_ssl::calculateHashSum(unsigned char **digest,const char *word){
    *digest = new unsigned char[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char *>(word), defaultWordLength, *digest);
}

void SHA1_ssl::setDefaultWordLength(unsigned int defaultLength) {
    defaultWordLength = defaultLength;
}

unsigned int SHA1_ssl::getDigestLength() {
    return SHA_DIGEST_LENGTH;
}