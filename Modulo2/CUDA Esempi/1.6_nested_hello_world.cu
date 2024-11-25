#include <stdio.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void nestedHelloWorld(int const iSize, int iDepth) {
    // Get the thread ID within the current block
    int tid = threadIdx.x;

    // Print a message with thread information and recursion depth
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    // Termination condition: if there's only one thread, end recursion
    if (iSize == 1) return;

    // Calculate the number of threads for the next level (halves the thread count)
    int nthreads = iSize >> 1;

    // Only thread 0 launches a new kernel, if there are still threads to launch
    if (tid == 0 && nthreads > 0) {
        // Recursively launch a new kernel with half the threads
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        
        // Print the nested execution depth
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main() {
    // Initial number of threads
    const int nThreads = 8;
    
    // Initial recursion depth
    const int iDepth = 0;

    // Configure launch dimensions
    dim3 block(nThreads);
    dim3 grid(2);

    // Launch the kernel
    nestedHelloWorld<<<grid, block>>>(nThreads, iDepth);

    // Synchronize to ensure all kernels are completed
    CHECK(cudaDeviceSynchronize());

    // Check for any errors
    CHECK(cudaGetLastError());

    // Clean up and exit
    CHECK(cudaDeviceReset());

    return 0;
}
