#include <iostream>
#include <cstring>
#include "MD5KernelHeader.cuh"
#include "../cudaUtils.cuh"

unsigned int calculateWorkingBufferLength(unsigned int wordLength);

unsigned char hexToInt(unsigned char a, unsigned char b);

void parseInput(int argc, char **argv, int *length, unsigned char *digest);

float startKernel(int length, char *word_gpu, unsigned char *digest_gpu, bool *kernel_end_gpu);

int main(int argc, char **argv) {
    unsigned char digest[DIGEST_LENGTH];
    int length;
    parseInput(argc, argv, &length, digest);

    char *word = new char[length + 1];
    char *word_gpu = allocateArrayOnGPU<char>(length);

    unsigned char *digest_gpu = allocateArrayOnGPU<unsigned char>(DIGEST_LENGTH);
    transferDataToGPU(digest_gpu, digest, DIGEST_LENGTH);

    bool *kernel_end_gpu = allocateArrayOnGPU<bool>(1);
    bool kernel_end = false;
    transferDataToGPU(kernel_end_gpu, &kernel_end, 1);

    float kernelDuration = startKernel(length, word_gpu, digest_gpu, kernel_end_gpu);

    transferDataFromGPU(word, word_gpu, length);
    word[length] = '\0';

    transferDataFromGPU(&kernel_end, kernel_end_gpu, 1);

    std::cout << (kernel_end ? word : "-") << std::endl;
    printf("Time to generate: %.2f ms \n", kernelDuration);

    freeArrayGPU(word_gpu);
    freeArrayGPU(digest_gpu);
    delete[] word;

    return 0;
}

float startKernel(int length, char *word_gpu, unsigned char *digest_gpu, bool *kernel_end_gpu) {
    //todo check this calculation
    int workingBufferLength = calculateWorkingBufferLength(length);
    unsigned long threadNum = 128;
    unsigned long blockNum = 1;
    for (int i = 0; i < length && i < 4; ++i) {
        blockNum *= 256;
    }
    blockNum /= threadNum;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calculateHashSum <<< blockNum, threadNum >>>(digest_gpu, word_gpu, workingBufferLength, length, kernel_end_gpu);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    return time;
}

void parseInput(int argc, char **argv, int *length, unsigned char *digest) {
    if (argc != 3) {
        exit(EXIT_FAILURE);
    }
    *length = atoi(argv[1]);

    char digest_hex[DIGEST_LENGTH * 2 + 1];
    strcpy(reinterpret_cast<char *>(&digest_hex), argv[2]);

    for (int i = 0; i < DIGEST_LENGTH; i++) {
        digest[i] = hexToInt(digest_hex[2 * i], digest_hex[2 * i + 1]);
    }
}

unsigned char hexToInt(unsigned char a, unsigned char b) {
    a = a - '0' < 10 ? a - '0' : a - 'a' + 10;
    b = b - '0' < 10 ? b - '0' : b - 'a' + 10;
    return (a * 16) + b;
}

unsigned int calculateWorkingBufferLength(unsigned int wordLength) {
    unsigned int toAdd = 64 - (wordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return wordLength + toAdd + 8;
}
