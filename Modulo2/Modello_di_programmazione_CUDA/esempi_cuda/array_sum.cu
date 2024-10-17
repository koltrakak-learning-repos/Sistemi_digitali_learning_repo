#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        /*
            & operatore per bitwise and.
            facendo:
                rand() & 0xFF
            si sta limitando rand() ai valori tra 0 e 255

            (più veloce rispetto a rand() % 256)
        */
        ip[i] = (float)(rand() & 0xFF) / 10.0f; // [0.0, 25.5]
    }
}

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N) {
    //calcolo indice globale
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //non considero gli eventuali thread in eccesso nell'ultimo blocco
    if (i < N)
        C[i] = A[i] + B[i];
}


int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <array_size> <block_size>\n", argv[0]);
        return 1;
    }

    // Parse array size and block size from command line
    int nElem = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    printf("Vector size: %d\n", nElem);
    printf("Block size: %d\n", blockSize);

    size_t nBytes = nElem * sizeof(float);

    // Allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);  //risultato host
    gpuRef = (float *)malloc(nBytes);   //risultato copiato dal device nella memoria dell'host

    // Initialize data
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Add vector on host
    double iStart = cpuSecond();
    sumArrayOnHost(h_A, h_B, hostRef, nElem);
    double iElaps = cpuSecond() - iStart;
    printf("\tsumArrayOnHost Time elapsed %f sec\n", iElaps);

    // Allocate device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // Transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Invoke kernel
    int gridSize = (nElem + blockSize - 1) / blockSize;

    iStart = cpuSecond();
    sumArrayOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("\tsumArrayOnGPU <<<%d, %d>>> Time elapsed %f sec\n", gridSize, blockSize, iElaps);

    // Copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Check device results
    checkResult(hostRef, gpuRef, nElem);

    // Free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
