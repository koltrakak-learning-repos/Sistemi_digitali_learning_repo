#include <cuda_runtime.h>
#include <stdio.h>

/**
 * This kernel demonstrates thread-level divergence
 * Each adjacent thread takes a different path through the if-else statement
 * This causes maximum warp divergence as threads within the same warp
 * will need to execute both paths sequentially
 */
__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f, b = 0.0f;
    if (tid % 2 == 0) a = 100.0f;
    else b = 200.0f;
    c[tid] = a + b;
}

/**
 * This kernel demonstrates warp-level divergence
 * Threads are grouped by warps, so all threads within the same warp
 * take the same path through the if-else statement
 * This minimizes warp divergence and should show better performance
 */
__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f, b = 0.0f;
    if ((tid / warpSize) % 2 == 0) a = 100.0f;
    else b = 200.0f;
    c[tid] = a + b;
}

int main() {
    // Configuration parameters
    int numThreads = 1024 * 1024;  // Launch 1M threads
    int threadsPerBlock = 256;     // Use 256 threads per block
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    // Device memory allocation
    float *d_c;
    cudaMalloc((void**)&d_c, numThreads * sizeof(float));
    
    // Launch configuration information
    printf("Launching kernels with %d threads (%d blocks, %d threads/block)\n", 
           numThreads, numBlocks, threadsPerBlock);
    
    // Launch kernel with thread-level divergence
    // Profile this kernel to observe poor branch efficiency
    mathKernel1<<<numBlocks, threadsPerBlock>>>(d_c);
    cudaDeviceSynchronize();
    
    // Launch kernel with warp-level divergence
    // Profile this kernel to observe better branch efficiency
    mathKernel2<<<numBlocks, threadsPerBlock>>>(d_c);
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_c);
    
    return 0;
}