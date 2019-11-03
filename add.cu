#include <iostream>
#include <math.h>
#include <ctime>
#include "cuda_clion_hack.hpp"

const int BLOCK_NUMB = 1;
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y, bool *processed) {
    int blockSize = n / BLOCK_NUMB;
    for (int i = blockIdx.x * blockSize; i < (blockIdx.x + 1) * blockSize; i++) {
        y[i] = x[i] + y[i];
        y[i] = 2 * y[i];
        x[i] = 3 * x[i];
        x[i] = x[i] + y[i];
        y[i] = x[i] * 5;
        processed[i] = true;
    }
}

int main(void) {
    const int N = 1 << 20;
    float *x, *y;
    bool *p;
    clock_t start_time, end_time;
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&p, N * sizeof(bool));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        p[i] = false;
    }
    start_time = clock();
    // Run kernel on 1M elements on the GPU
    add <<< BLOCK_NUMB, 1 >>> (N, x, y, p);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    end_time = clock();
    std::cout << "end GPU work" << std::endl
              << "time: " << end_time - start_time << std::endl;
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    float error;
    float result = 45.0f;
    for (int i = 0; i < N; i++) {
        error = fabs(y[i] - result);
        if (!p[i])
            std::cout << i << ": " << error << std::endl;
        maxError = fmax(maxError, error);
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}