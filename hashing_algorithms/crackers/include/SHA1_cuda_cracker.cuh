// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_SHA1CUDACRACKER_CUH
#define INYNIERKA_SHA1CUDACRACKER_CUH

#include "../../../cuda_clion_hack.hpp"

#define DIGEST_LENGTH 20

__global__ void
calculateHashSum(unsigned char *digest, char *message, int workingBufferLength, int lenght);


#endif //INYNIERKA_SHA1CUDACRACKER_CUH
