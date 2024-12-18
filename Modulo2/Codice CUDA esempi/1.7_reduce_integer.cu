/**
* CUDA Parallel Reduction Implementations
* 
* This program compares different parallel reduction algorithms:
* 1. Sequential CPU reduction (iterative and recursive)
* 2. Basic GPU reduction with neighboring pairs
* 3. GPU reduction with reduced divergence 
* 4. GPU reduction with interleaved pairs
* 5. GPU reduction with 2x unrolling
* 6. GPU reduction with 4x unrolling
* 7. GPU reduction with 8x unrolling
* 8. GPU reduction with warp-level optimizations + 8x unrolling 
* 9. GPU reduction with complete unrolling + warp optimizations + 8x unrolling
* 10. Template-based GPU reduction with compile-time optimizations
* 11. Template-based GPU reduction using shared memory
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * CPU iterative reduction function
 * Simple sequential sum
 */
int iterativeReduce(int *data, int size) {
    int sum = 0;
    for(int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * CPU recursive reduction function
 * Performs in-place reduction by recursively summing adjacent elements
 */
int recursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    return recursiveReduce(data, stride);
}

/**
 * High-precision timer function
 */
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/**
 * GPU Kernel 1: Basic neighboring reduction
 */
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    
    if (tid >= n) return;
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


/**
 * GPU Kernel 2: Optimized reduction with less divergence
 * Uses more efficient indexing to reduce thread divergence
 */
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID and global index
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    
    // Boundary check
    if (idx >= n) return;
    
    // In-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // Convert tid into local array index
        int index = 2 * stride * tid;
        
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        
        // Synchronize within threadblock
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 3: Interleaved Pair Implementation
 * Uses a different striding pattern to reduce memory conflicts
 */
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    
    // Boundary check
    if(idx >= n) return;
    
    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 4: Unrolling-2 Implementation
 * Process two elements per thread to improve performance
 */
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    
    // Unrolling 2 data blocks
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();
    
    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 5: Unrolling4 Implementation
 * Processes 4 elements per thread to improve performance
 */
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;
    
    // Unrolling 4 data blocks
    if (idx + 3*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();
    
    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 6: Unrolling8 Implementation
 * Processes 8 elements per thread to improve performance
 */
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    // Unrolling 8 data blocks
    if (idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int b1 = g_idata[idx + 4*blockDim.x];
        int b2 = g_idata[idx + 5*blockDim.x];
        int b3 = g_idata[idx + 6*blockDim.x];
        int b4 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    
    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 7: Unrolling with Warp-level Optimization
 * Processes 8 elements per thread and uses warp-level optimizations
 */
__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    // Unrolling 8 data elements per thread
    if (idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int b1 = g_idata[idx + 4*blockDim.x];
        int b2 = g_idata[idx + 5*blockDim.x];
        int b3 = g_idata[idx + 6*blockDim.x];
        int b4 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    
    // In-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    // Unrolling last warp (no sync needed - warp synchronous)
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * GPU Kernel 8: Complete Unrolling with Warp Optimizations
 * Combines 8-element unrolling with completely unrolled reduction
 */
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    // Unrolling 8 data elements per thread
    if (idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    
    // Complete unrolling of the reduction
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();
    
    // Unrolling warp (no sync needed - warp synchronous)
    if (tid < 32) {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


/**
 * GPU Kernel 9: Template-based Complete Unrolling
 * Uses template parameter for block size to enable compile-time optimizations
 */
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n) {
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    // Convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    
    // Unrolling 8 data elements per thread
    if (idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    
    // In-place reduction with complete unrolling
    if (iBlockSize >= 1024 && tid < 512) {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();
    
    if (iBlockSize >= 512 && tid < 256) {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();
    
    if (iBlockSize >= 256 && tid < 128) {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();
    
    if (iBlockSize >= 128 && tid < 64) {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();
    
    // Unrolling warp (no sync needed - warp synchronous)
    if (tid < 32) {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    // Write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

template __global__ void reduceCompleteUnroll<1024>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnroll<512>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnroll<256>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnroll<128>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnroll<64>(int *g_idata, int *g_odata, unsigned int n);


template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrollShared(int *g_idata, int *g_odata, unsigned int n) {
    // Shared memory per il blocco
    __shared__ int smem[iBlockSize];
    
    // Set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    // Azzera la shared memory per sicurezza
    int tmp = 0;
    
    // Carica i dati dalla global memory alla shared memory
    // Unrolling 8 data elements per thread
    if (idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        tmp = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    smem[tid] = tmp; 
    __syncthreads();
    
    // In-place reduction in shared memory
    if (iBlockSize >= 1024 && tid < 512) {
        smem[tid] += smem[tid + 512];
    }
    __syncthreads();
    
    if (iBlockSize >= 512 && tid < 256) {
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();
    
    if (iBlockSize >= 256 && tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();
    
    if (iBlockSize >= 128 && tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();
    
    // Unrolling warp (no sync needed - warp synchronous)
    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    // Write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = smem[0];
    }
}

// Dichiarazione esplicita del template
template __global__ void reduceCompleteUnrollShared<1024>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnrollShared<512>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnrollShared<256>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnrollShared<128>(int *g_idata, int *g_odata, unsigned int n);
template __global__ void reduceCompleteUnrollShared<64>(int *g_idata, int *g_odata, unsigned int n);

int main(int argc, char **argv) {
    // Device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction on device %d: %s\n", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    // Initialize problem size
    int size = 1 << 26;  // Total number of elements to reduce
    printf("Array size: %d\n", size);

    // Set execution configuration
    int blocksize = 512;  // Default block size
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Grid size: %d, Block size: %d\n", grid.x, block.x);

    // Allocate and initialize host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // Initialize input array with random values (max 255)
    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() % 10); //(int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    // Allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    // Variables for timing and results
    double gpu_start;
    int gpu_sum = 0;
    
    // 0. CPU Reduction (baseline)
    double cpu_start = cpuSecond();
    int cpu_sum = iterativeReduce(tmp, size);
    double cpu_elaps_0 = cpuSecond() - cpu_start;
    printf("\n1. CPU iterative reduction time: %.6f seconds, Sum: %d\n", cpu_elaps_0, cpu_sum);

    // 1. CPU Reduction (baseline)
    cpu_start = cpuSecond();
    cpu_sum = recursiveReduce(tmp, size);
    double cpu_elaps_1 = cpuSecond() - cpu_start;
    printf("\n1. CPU recursive reduction time: %.6f seconds, Sum: %d\n", cpu_elaps_1, cpu_sum);

    // 2. Basic GPU Reduction (neighboring approach)
    double gpu_elaps_2;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();

    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    gpu_elaps_2 = cpuSecond() - gpu_start;
    printf("2. Basic GPU reduction time: %.6f seconds, Sum: %d\n", gpu_elaps_2, gpu_sum);

    // 3. Optimized GPU Reduction (less divergence)
    double gpu_elaps_3;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    gpu_elaps_3 = cpuSecond() - gpu_start;
    printf("3. Less Divergent GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_3, gpu_sum);

    // 4. Interleaved GPU Reduction
    double gpu_elaps_4;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    gpu_elaps_4 = cpuSecond() - gpu_start;
    printf("4. Interleaved GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_4, gpu_sum);

    // 5. Unrolling2 GPU Reduction
    double gpu_elaps_5;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/2 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/2; i++) gpu_sum += h_odata[i];
    gpu_elaps_5 = cpuSecond() - gpu_start;
    printf("5. Unrolling2 GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_5, gpu_sum);

    // 6 Unrolling4 GPU Reduction
    double gpu_elaps_6;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    gpu_start = cpuSecond();
    reduceUnrolling4<<<grid.x/4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];
    gpu_elaps_6 = cpuSecond() - gpu_start;
    printf("6 Unrolling4 GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_6, gpu_sum);

    // 7 Unrolling8 GPU Reduction
    double gpu_elaps_7;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    gpu_start = cpuSecond();
    reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    gpu_elaps_7 = cpuSecond() - gpu_start;
    printf("7 Unrolling8 GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_7, gpu_sum);

    // 8. UnrollWarps8 GPU Reduction
    double gpu_elaps_8;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    gpu_elaps_8 = cpuSecond() - gpu_start;
    printf("8. UnrollWarps8 GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_8, gpu_sum);

    // 9. CompleteUnrollWarps8 GPU Reduction
    double gpu_elaps_9;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    gpu_elaps_9 = cpuSecond() - gpu_start;
    printf("9. CompleteUnrollWarps8 GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_9, gpu_sum);

    // 10. Template-based Complete Unroll Reduction
    double gpu_elaps_10;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    switch (blocksize) {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 64:
            reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
    }
    cudaDeviceSynchronize();
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    gpu_elaps_10 = cpuSecond() - gpu_start;
    printf("10. Template Complete Unroll GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_10, gpu_sum);

    // 11. Template-based Complete Unroll Reduction
    double gpu_elaps_11;
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_start = cpuSecond();
    switch (blocksize) {
        case 1024:
            reduceCompleteUnrollShared<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnrollShared<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnrollShared<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnrollShared<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 64:
            reduceCompleteUnrollShared<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
    }
    cudaDeviceSynchronize();
    gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    gpu_elaps_11 = cpuSecond() - gpu_start;
    printf("11. Template Complete Unroll (SMEM) GPU time: %.6f seconds, Sum: %d\n", gpu_elaps_11, gpu_sum);


    // Final Performance Analysis
    printf("\nComplete Performance Analysis:\n");
    printf("0. CPU Iterative Base time: %.6f seconds (baseline)\n", cpu_elaps_0);
    printf("1. CPU Recursive Base time: %.6f seconds (baseline)\n", cpu_elaps_1);
    printf("2. Basic GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_2);
    printf("3. Less Divergent GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_3);
    printf("4. Interleaved GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_4);
    printf("5. Unrolling2 GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_5);
    printf("6. Unrolling4 GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_6);
    printf("7. Unrolling8 GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_7);
    printf("8. UnrollWarps8 GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_8);
    printf("9. CompleteUnrollWarps8 GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_9);
    printf("10. Template Complete Unroll GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_10);
    printf("11. Template Complete Unroll (SMEM) GPU Speedup: %.2fx\n", cpu_elaps_1 / gpu_elaps_11);

    // Cleanup
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();

    // Verify results
    bool bResult = (gpu_sum == cpu_sum);
    if (!bResult) {
        printf("\nTest failed! GPU sum does not match CPU sum\n");
    } else {
        printf("\nTest passed!\n");
    }

    return EXIT_SUCCESS;
}

