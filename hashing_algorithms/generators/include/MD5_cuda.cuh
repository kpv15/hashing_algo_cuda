#ifndef INYNIERKA_MD5_CUDA_CUH
#define INYNIERKA_MD5_CUDA_CUH

#include "../../../cuda_clion_hack.hpp"

namespace MD5_cuda {

    __global__ void calculateHashSum(unsigned char *digest, char *word, unsigned long int workingBufferLength,
                                     unsigned long int wordLength, unsigned long int n);
}
#endif //INYNIERKA_MD5_CUDA_CUH
