#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>


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

#define CHECK_CUFFT(call) \
{ \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

/*
    funzioni di utilità
*/

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}


int main(int argc, char **argv) {
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set-up dei campioni fittizzi
    const size_t N = 1 << 23; // Array size: 2^23
    const size_t bytes = N * sizeof(cufftComplex);  // sono due float32

    cufftComplex *h_data = (cufftComplex*)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_data[i].x = (float)(i % 256); // Real part
        h_data[i].y = 0.0f;            // Imaginary part
    }

    // Allocate device memory
    cufftComplex *d_data;
    CHECK(cudaMalloc((void**)&d_data, bytes));
    // Copy data to device
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    double start = cpuSecond();
    // Create cuFFT plan
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    // Execute FFT on the device
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    // Copy result back to host
    CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    double elapsed = cpuSecond() - start;

    // Clean up
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK(cudaFree(d_data));
    free(h_data);

    printf("tempo: %f ms\n", elapsed*1000);
}