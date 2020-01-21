//
// Created by grzegorz on 20.01.2020.
//

#ifndef INYNIERKA_SHA1_CUDA_CUH
#define INYNIERKA_SHA1_CUDA_CUH

#include "../../../cuda_clion_hack.hpp"

namespace SHA1_cuda{
    __global__ void calculateHashSum(unsigned char *digest, const char *word, unsigned long int workingBufferLength,
                                     unsigned long int wordLength, unsigned long int n);
}

#endif //INYNIERKA_SHA1_CUDA_CUH
