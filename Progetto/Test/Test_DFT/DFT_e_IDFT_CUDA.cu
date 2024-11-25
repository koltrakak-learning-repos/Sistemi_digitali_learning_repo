#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SAMPLE_RATE 44100
#define PI 3.14159265358979323846

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

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

typedef struct {
    double real;
    double imag;
} complex;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}




int main(int argc, char **argv) {
    // Configuration parameters
    const char* FILE_NAME = "StarWars3_44100.wav";
    
    int numThreads = 1024 * 1024;  // Launch 1M threads
    int threadsPerBlock = 256;     // Use 256 threads per block
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    printf("%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
 
    // Parsing del file in input
    drwav wav_in;    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file A440.wav.\n");
        return 1;
    }
    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;
    printf("NUMERO DI CAMPIONI NEL FILE AUDIO SCELTO: %ld; -> %0.2f secondi\n\n", num_samples, (float)num_samples/SAMPLE_RATE);

    // Allocazione del buffer per i dati audio (PCM a 16 bit)
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Allocazione del buffer per le sinusoidi della DFT (N/2)
    complex* dft_samples = (complex*)malloc( (num_samples/2) * sizeof(complex));
    if (dft_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Leggi i campioni dal file in input
    size_t samples_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, signal_samples);
    if (samples_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        return 1;
    }
    drwav_uninit(&wav_in); 

    // Alloca memoria per i campioni sul device
    short* device_signal_samples;
    complex* device_dft_samples
    
    CHECK(cudaMalloc((short**)&device_signal_samples, num_samples*sizeof(short)));
    CHECK(cudaMalloc((complex**)&device_dft_samples, (num_samples/2)*sizeof(complex)));
    
    CHECK(cudaMemcpy(device_signal_samples, signal_samples, num_samples*sizeof(short), cudaMemcpyHostToDevice));





    // TODO: continualo...




    // Invoke kernel
    int gridSize = (nElem + blockSize - 1) / blockSize;

    iStart = cpuSecond();
    sumArrayOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumArrayOnGPU <<<%d, %d>>> Time elapsed %f sec\n", gridSize, blockSize, iElaps);

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
