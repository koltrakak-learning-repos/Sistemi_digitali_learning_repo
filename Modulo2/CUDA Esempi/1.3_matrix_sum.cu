#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to check CUDA errors
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

// Function to measure time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// Initialize matrix data with random values
void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Check if CPU and GPU results match
void checkResult(float *hostRef, float *gpuRef, const int size) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}


// CUDA kernel for matrix addition (2D grid and 2D block)
// This is the generic kernel that can be used for all configurations
// by adjusting the launch parameters in the main function
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int W, int H) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < W && iy < H){
        unsigned int idx = iy * W + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }    
}

// The following kernels are included for educational purposes only.
// They demonstrate how different grid/block configurations can be handled.
// In practice, the generic sumMatrixOnGPU2D kernel above can be used for all cases
// by adjusting the launch configuration.


// CUDA kernel for matrix addition (1D grid and 2D block)
// NOTE: This kernel is included for educational purposes.
// The same result can be achieved using sumMatrixOnGPU2D with the following configuration:
// block = dim3(1, block_dim_y);
// grid = dim3(W, 1);
// sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
__global__ void sumMatrixOnGPU1D2D(float *MatA, float *MatB, float *MatC, int W, int H) {
   unsigned int ix = blockIdx.x;
   unsigned int iy = threadIdx.y;
   if (ix < W && iy < H) {
       unsigned int idx = iy * W + ix;
       MatC[idx] = MatA[idx] + MatB[idx];
   }
}

// CUDA kernel for matrix addition (2D grid and 1D block)
// NOTE: This kernel is included for educational purposes.
// The same result can be achieved using sumMatrixOnGPU2D with the following configuration:
// block = dim3(block_dim_x, 1);
// grid = dim3((W + block.x - 1) / block.x, H);
// sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
__global__ void sumMatrixOnGPU2D1D(float *MatA, float *MatB, float *MatC, int W, int H) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * W + ix;
    if (ix < W && iy < H)
        MatC[idx] = MatA[idx] + MatB[idx];
}


// CUDA kernel for matrix addition (1D grid and 1D block)
// NOTE: This kernel uses a different approach with a for loop.
// It cannot be directly replaced by sumMatrixOnGPU2D without modifying the kernel logic.
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int W, int H) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < W ) {
        for (int iy = 0; iy < H; iy++) {
            int idx = iy * W + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

// Host function for matrix addition
void sumMatrixOnHost(float *MatA, float *MatB, float *MatC, int W, int H) {
  for (int i = 0; i < H; i++) {  
      for (int j = 0; j < W; j++) { 
          int idx = i * W + j; 
          MatC[idx] = MatA[idx] + MatB[idx]; 
      }
  }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // Set CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Check command line arguments
    if (argc != 6) {
        printf("Usage: %s <kernel_type> <block_dim_x> <block_dim_y> <matrix_width> <matrix_height>\n", argv[0]);
        printf("Kernel types: CPU, 2D, 1D2D, 1D, 2D1D\n");
        return 1;
    }

    // Parse command line arguments
    const char* kernel_type = argv[1];
    int block_dim_x = atoi(argv[2]);
    int block_dim_y = atoi(argv[3]);
    int W = atoi(argv[4]);  // Matrix width
    int H = atoi(argv[5]);  // Matrix height

    int size = W * H;
    int nBytes = size * sizeof(float);
    printf("Matrix size: W %d H %d\n", W, H);

    // Allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Initialize matrix data
    double iStart = cpuSecond();
    initialData(h_A, size);
    initialData(h_B, size);
    double iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Add matrix on CPU
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, W, H);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnHost elapsed %f s\n", iElaps);

    // If CPU execution is requested, we're done
    if (strcmp(kernel_type, "CPU") == 0) {
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
        return 0;
    }

    // Allocate device memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions 
    dim3 block, grid;
    block = dim3(block_dim_x, block_dim_y);
    grid = dim3((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // Launch kernel
    iStart = cpuSecond();
    if (strcmp(kernel_type, "2D") == 0) {
        sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    } else if (strcmp(kernel_type, "1D2D") == 0) {
        block = dim3(1, block_dim_y);
        grid = dim3(W);
        sumMatrixOnGPU1D2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    } else if (strcmp(kernel_type, "1D") == 0) {
        block = dim3(block_dim_x);
        grid = dim3((W + block.x - 1) / block.x);
        sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    } else if (strcmp(kernel_type, "2D1D") == 0) {
        block = dim3(block_dim_x);
        // grid remains the same as in the 2D case
        sumMatrixOnGPU2D1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, W, H);
    } else {
        printf("Invalid kernel type\n");
        return 1;
    }
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("%s <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n",
           kernel_type, grid.x, grid.y, block.x, block.y, iElaps);

    // Copy kernel result from device to host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // Check result
    checkResult(hostRef, gpuRef, size);

    // Free device memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset CUDA device
    CHECK(cudaDeviceReset());

    return 0;
}
