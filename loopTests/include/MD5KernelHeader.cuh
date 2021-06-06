// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDACRACKER_CUH
#define INYNIERKA_MD5CUDACRACKER_CUH

#define DIGEST_LENGTH 16

__global__ void
calculateHashSum(unsigned char *digest, char *message, int workingBufferLength, int lenght, volatile bool *kernel_end);


#endif //INYNIERKA_MD5CUDACRACKER_CUH
