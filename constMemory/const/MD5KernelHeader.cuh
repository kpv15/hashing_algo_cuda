// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_MD5CUDACRACKER_CUH
#define INYNIERKA_MD5CUDACRACKER_CUH

#define DIGEST_LENGTH 16

extern __constant__ unsigned char DIGEST[DIGEST_LENGTH];
extern __constant__ int WORKING_BUFFER_LENGTH;
extern __constant__ int LENGTH;

__global__ void
calculateHashSum(char *word, volatile bool *kernel_end);


#endif //INYNIERKA_MD5CUDACRACKER_CUH
