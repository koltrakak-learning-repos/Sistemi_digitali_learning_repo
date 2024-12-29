#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NUM_STREAMS 8

const int N = 1 << 20; 

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    cudaStream_t streams[NUM_STREAMS];
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    float *h_a, *h_b, *h_c[NUM_STREAMS]; 
    
    // Allocazione pinned memory per host
    cudaMallocHost(&h_a, N * sizeof(float));
    cudaMallocHost(&h_b, N * sizeof(float));
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocHost(&h_c[i], N * sizeof(float));
    }
    
    // Inizializzazione dati
    for(int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Creazione stream e allocazione memoria device
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_a[i], N * sizeof(float));
        cudaMalloc(&d_b[i], N * sizeof(float));
        cudaMalloc(&d_c[i], N * sizeof(float));
        
        // Copia asincrona dei dati di input
        cudaMemcpyAsync(d_a[i], h_a, N * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b, N * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Lancio del Kernel
        vectorAdd<<<1, 64, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);
        
        // Copia asincrona dei risultati
        cudaMemcpyAsync(h_c[i], d_c[i], N * sizeof(float), 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Sincronizzazione di tutti gli stream
    cudaDeviceSynchronize();

    // Free
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaFreeHost(h_c[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    return 0;
}
