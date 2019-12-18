#include "../../cuda_clion_hack.hpp"

__global__ void calculateHashSum(unsigned char **&digest, char **&word, unsigned long int workingBufferLength,
                                 unsigned long int wordLength);