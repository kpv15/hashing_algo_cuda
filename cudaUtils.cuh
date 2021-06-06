#ifndef PRIR_CUDAUTILS_CUH
#define PRIR_CUDAUTILS_CUH

unsigned int getBlocksNumber(const unsigned int threadsNum, const unsigned int elementsCount) {
    return ceil(elementsCount / threadsNum) + 1;
}

void checkError(cudaError_t returnCode, std::string exceptionMessage) {
    if (returnCode != cudaSuccess) {
        std::cout << exceptionMessage << ": " << cudaGetErrorName(returnCode) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void synchronizeKernel() {
    checkError(cudaDeviceSynchronize(), "error during Device Synchronize");
}

template<typename T>
void transferDataFromGPU(T *arrayOnCPU, T *arrayOnGPU, size_t elementNumber) {
    cudaError_t errorCode = cudaMemcpy(arrayOnCPU, arrayOnGPU, sizeof(T) * elementNumber, cudaMemcpyDeviceToHost);
    checkError(errorCode, "error during transfer data from gpu");
}

template<typename T>
void transferDataToGPU(T *arrayOnGPU, T *arrayOnCPU, size_t elementNumber) {
    cudaError_t errorCode = cudaMemcpy(arrayOnGPU, arrayOnCPU, sizeof(T) * elementNumber, cudaMemcpyHostToDevice);
    checkError(errorCode, "error during transfer data to gpu");
}

template<typename T>
T *allocateArrayOnGPU(const size_t elementsNumber) {
    T *table_addr;
    cudaError_t errorCode = cudaMalloc((void **) &table_addr, elementsNumber * sizeof(T));
    checkError(errorCode, "error during alloc memory on GPU, error code");
    return table_addr;
}

template<typename T>
void freeArrayGPU(T *table_addr) {
    checkError(cudaFree(table_addr), "error during free memory on GPU, error code");
}

#endif //PRIR_CUDAUTILS_CUH
