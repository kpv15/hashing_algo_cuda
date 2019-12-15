//
// Created by grzegorz on 06.11.2019.
//

#include "include/MD5_cuda.cuh"

void MD5_cuda::setDefaultWordLength(unsigned int i) {
    this->defaultWordLength = i;
}

unsigned int MD5_cuda::getDigestLength() {
    return DIGEST_LENGTH;
}

unsigned int MD5_cuda::calculateWorkingBufferLength() {
    unsigned int toAdd = 64 - (defaultWordLength + 8) % 64;
    if (toAdd == 0) toAdd = 64;
    return defaultWordLength + toAdd + 8;
}

void MD5_cuda::createWorkingBuffer(const char *word) {
    unsigned long int calculatedWorkingBufferLength = calculateWorkingBufferLength();
    if (workingBuffer != nullptr && calculatedWorkingBufferLength != workingBufferLength)
        delete[] workingBuffer;
    if (workingBuffer == nullptr)
        workingBuffer = new unsigned char[calculatedWorkingBufferLength];
    workingBufferLength = calculatedWorkingBufferLength;
    numberOfChunks = workingBufferLength / 64;

    std::memcpy(workingBuffer, word, defaultWordLength);
    workingBuffer[defaultWordLength] = 0b10000000;
    std::memset(workingBuffer + defaultWordLength + 1, 0, workingBufferLength - defaultWordLength - 1 - 8);
    std::memcpy(workingBuffer + workingBufferLength - 8, &defaultWordLength, 8);
}

void MD5_cuda::createDigestBuffer() {
    if (digestBuffer == nullptr)
        digestBuffer = new unsigned char[DIGEST_LENGTH];
    memcpy(digestBuffer, DEFAULT_DIGEST_BUFFER, DIGEST_LENGTH);
}

unsigned const int MD5_cuda::DEFAULT_DIGEST_BUFFER[4] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476
};

MD5_cuda::~MD5_cuda() {
    delete[] digestBuffer;
    delete[] workingBuffer;
}
