//
// Created by grzegorz on 12.01.2020.
//

#include <iostream>
#include <cstring>
#include <chrono>
#include <openssl/sha.h>

#define DIGEST_LENGTH 20

#define MAX_MESSAGE_LENGTH 100

int crack(int min_length, int max_length, unsigned char *digest);

inline unsigned char hexToInt(unsigned char a, unsigned char b) {
    a = a - '0' < 10 ? a - '0' : a - 'a' + 10;
    b = b - '0' < 10 ? b - '0' : b - 'a' + 10;
    return (a * 16) + b;
}

int main(int argc, char **argv) {

    char digest_hex[DIGEST_LENGTH * 2 + 1];
    unsigned char digest[DIGEST_LENGTH];
    int min = 0;
    int max = 0;
    if (argc >= 4) {
        min = atoi(argv[1]);
        max = atoi(argv[2]);
        strcpy(reinterpret_cast<char *>(&digest_hex), argv[3]);
    }

    for (int i = 0; i < DIGEST_LENGTH; i++) {
        digest[i] = hexToInt(digest_hex[2 * i], digest_hex[2 * i + 1]);
    }

    crack(min, max, digest);

}

void calculateHashSum(unsigned char *digest, char *word, int lenght){
    unsigned char workingBuffer[MAX_MESSAGE_LENGTH];
    unsigned char step_digest[DIGEST_LENGTH];
    memset(workingBuffer, 0, MAX_MESSAGE_LENGTH);

    bool done;
    do {

        SHA1(reinterpret_cast<const unsigned char *>(workingBuffer), lenght, step_digest);
        if(memcmp(step_digest,digest,DIGEST_LENGTH)==0){
            memcpy(word,workingBuffer,lenght);
            break;
        }

        int i = 0;
        while (++workingBuffer[i] == 0 && i < lenght)
            i++;
        done = true;
        for (int i = 0; i < lenght; i++) {
            if (workingBuffer[i] != 0) {
                done = false;
                break;
            }
        }
    }while (!done);
}

int crack(int min_length, int max_length, unsigned char *digest) {

    char *word = new char[max_length + 1];

    for (int length = min_length; length <= max_length; length++) {

        std::cout << "checking word with length: " << length << std::endl;
        auto startKernel = std::chrono::high_resolution_clock::now();

//        auto startKernel = std::chrono::high_resolution_clock::now();
        calculateHashSum(digest,word,length);

        auto stopKernel = std::chrono::high_resolution_clock::now();

        word[length] = '\0';

        auto durationKernel = std::chrono::duration_cast<std::chrono::microseconds>(stopKernel - startKernel);

        std::cout << word << "\tin: " << durationKernel.count() << std::endl;
    }

    delete[]word;

    return 0;
}