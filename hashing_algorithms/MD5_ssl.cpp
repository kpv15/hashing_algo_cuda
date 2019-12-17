//
// Created by grzegorz on 09.11.2019.
//

#include "include/MD5_ssl.h"
#include <openssl/md5.h>

void MD5_ssl::calculateHashSum(unsigned char **digest,const char *word){
    *digest = new unsigned char[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char *>(word), defaultWordLength, *digest);
}

void MD5_ssl::setDefaultWordLength(unsigned int defaultLength) {
    defaultWordLength = defaultLength;
}

unsigned int MD5_ssl::getDigestLength() {
    return MD5_DIGEST_LENGTH;
}
