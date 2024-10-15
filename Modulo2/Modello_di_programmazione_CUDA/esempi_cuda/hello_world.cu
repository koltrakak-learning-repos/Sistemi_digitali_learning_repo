#include <stdio.h>

__global__ void helloFromGPU()
{
   printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main()
{
   // Launch the kernel
   helloFromGPU<<<1, 10>>>();

   // Wait for the GPU to finish
   cudaDeviceSynchronize();

   return 0;
}
