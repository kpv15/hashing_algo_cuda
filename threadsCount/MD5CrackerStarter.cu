#include <iostream>
#include <cstring>
#include "MD5KernelHeader.cuh"
#include "../cudaUtils.cuh"

unsigned int calculateWorkingBufferLength(unsigned int wordLength);

unsigned char hexToInt(unsigned char a, unsigned char b);

void parseInput(int argc, char **argv, int *length, int *threadsNum, unsigned char *digest);

float startKernel(int length, char *word_gpu, bool *kernel_end_gpu, int threadNum);

int main(int argc, char **argv) {
    unsigned char digest[DIGEST_LENGTH];
    int length;
    int threadsNum;
    parseInput(argc, argv, &length, &threadsNum, digest);

    char *word = new char[length + 1];
    char *word_gpu = allocateArrayOnGPU<char>(length);

    checkError(cudaMemcpyToSymbol(DIGEST, digest, sizeof(unsigned char) * DIGEST_LENGTH),
               "error during copy data to symbol DIGEST on GPU");

    bool *kernel_end_gpu = allocateArrayOnGPU<bool>(1);
    bool kernel_end = false;
    transferDataToGPU(kernel_end_gpu, &kernel_end, 1);

    float kernelDuration = startKernel(length, word_gpu, kernel_end_gpu, threadsNum);

    transferDataFromGPU(word, word_gpu, length);
    word[length] = '\0';

    transferDataFromGPU(&kernel_end, kernel_end_gpu, 1);

    std::cout << (kernel_end ? word : "-") << std::endl;
    printf("Time to generate: %.2f ms \n", kernelDuration);

    freeArrayGPU(word_gpu);
    delete[] word;

    return 0;
}

float startKernel(int length, char *word_gpu, bool *kernel_end_gpu, int threadNum) {
    //todo check this calculation
    int workingBufferLength = calculateWorkingBufferLength(length);
    checkError(cudaMemcpyToSymbol(WORKING_BUFFER_LENGTH, &workingBufferLength, sizeof(int)),
               "error during copy data to symbol WORKING_BUFFER_LENGTH on GPU");
    checkError(cudaMemcpyToSymbol(LENGTH, &length, sizeof(int)),
               "error during copy data to symbol LENGTH on GPU");
    unsigned long blockNum = 1;
    for (int i = 0; i < length && i < 4; ++i) {
        blockNum *= 256;
    }
    blockNum /= threadNum;
    std::cout << "threadNum: " << threadNum << " blockNum: " << blockNum << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calculateHashSum <<< blockNum, threadNum >>>(word_gpu, kernel_end_gpu);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    return time;
}

void parseInput(int argc, char **argv, int *length, int *threadsNum, unsigned char *digest) {
    if (argc != 4) {
        exit(EXIT_FAILURE);
    }
    *threadsNum = atoi(argv[1]);
    *length = atoi(argv[2]);

    char digest_hex[DIGEST_LENGTH * 2 + 1];
    strcpy(reinterpret_cast<char *>(&digest_hex), argv[3]);

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
