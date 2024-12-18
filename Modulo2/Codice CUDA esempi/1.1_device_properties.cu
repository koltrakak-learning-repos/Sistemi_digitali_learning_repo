#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);
    printf("CUDA System Information:\n");
    printf("CUDA Driver Version: %d.%d\n", CUDART_VERSION / 1000, CUDART_VERSION % 100);
    printf("CUDA Runtime Version: %d.%d\n", CUDART_VERSION / 1000, CUDART_VERSION % 100);
    printf("Number of CUDA Devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("1. Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("2. Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("3. Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("4. Clock Core: %d MHz\n", prop.clockRate / 1000);
        printf("5. Memory Clock: %d MHz\n", prop.memoryClockRate / 1000);
        printf("6. Memory Bus Width: %d bit\n", prop.memoryBusWidth);
        printf("7. L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("8. Shared Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
        printf("9. Maximum Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("10. Maximum Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("11. Maximum Block Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("12. Warp Size: %d\n", prop.warpSize);
        printf("13. Total Constant Memory: %d bytes\n", prop.totalConstMem);
        printf("14. Texture Alignment: %d bytes\n\n", prop.textureAlignment);
    }

    return 0;
}

