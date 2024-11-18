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

__global__ void reduceNeighbored(int *d_idata, int *d_odata, unsigned int n) {
    // ID del thread all'interno del blocco
    unsigned int tid = threadIdx.x; 
    // Indice globale del thread nella griglia e posizione del thread nel blocco dati di interesse  
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    int *idata = d_idata + blockIdx.x * blockDim.x; // Puntatore ai dati di input per questo blocco

    if (idx >= n)
        return; // Verifica se il thread è fuori dai limiti dei dati

    // Riduzione in-place nella memoria globale
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        // indice che assegna a thread ADIACENTI elementi che sono a distanza stride l'uno dall'altro.
        int index = 2 * stride * tid;   // ogni thread TID lavora su DUE elementi a distanza STRIDE

        // qua c'è comunque divergenza nelle ultime fasi di riduzione
        if (index < blockDim.x)
            idata[index] += idata[index + stride];
        
        __syncthreads();
    }

    if (tid == 0)
        d_odata[blockIdx.x] = idata[0]; // Il thread 0 scrive il risultato del blocco in g_odata
}

int main(int argc, char **argv) {
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    
    // Dimensione dell’array (potenza di 2)
    int size = 1 << 26;
    // Configurazione Griglia e Blocchi
    int blocksize = 512;
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    // Allocazione ed Inizializzazione Memoria Host
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    // Inizializzazione Random (max 10)
    for (int i = 0; i < size; i++)
        h_idata[i] = (int)(rand() % 10);

    // Allocazione Memoria Device
    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));
    // Trasferimento Dati Host -> Device + Calcolo Parallelo
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    // Sincronizzazione prima della Riduzione Globale
    cudaDeviceSynchronize(); 

    // Trasferimento Risultati Device -> Host + Somma Finale
    int gpu_sum = 0;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];

    printf("GPU Reduction Sum: %d\n", gpu_sum);
}