//
// Created by grzegorz on 09.11.2019.
//

#include "include/MD5_ssl.h"
#include <openssl/md5.h>

unsigned char *MD5_ssl::calculateHashSum(const char *word) {
    auto *result = new unsigned char[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char *>(word), defaultWordLength, result);

    return result;
}

void MD5_ssl::setDefaultWordLength(unsigned int defaultLength) {
    defaultWordLength = defaultLength;
}

unsigned int MD5_ssl::getDigestLength() {
    return MD5_DIGEST_LENGTH;
}
