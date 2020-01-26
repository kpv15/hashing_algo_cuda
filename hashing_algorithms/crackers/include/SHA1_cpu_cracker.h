// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_SHA1CPUCRACKER_CUH
#define INYNIERKA_SHA1CPUCRACKER_CUH

#include "../../../cuda_clion_hack.hpp"

#define DIGEST_LENGTH 20

void calculateHashSum(unsigned char *digest_g, char *message, int workingBufferLength, int lenght);


#endif //INYNIERKA_SHA1CPUCRACKER_CUH
