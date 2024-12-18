#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// CUDA compilation command:
// nvcc -D CUDA_FORCE_CDP1_IF_SUPPORTED -rdc=true 1.8_nested_reduce_integer.cu -o nested_reduce_integer
// -D CUDA_FORCE_CDP1_IF_SUPPORTED: forces CDP support despite deprecation
// -rdc=true: enables relocatable device code compilation, required for CDP


// Error checking macro
#define CHECK(call) \
{ \
   const cudaError_t error = call; \
   if (error != cudaSuccess) { \
       printf("Error: %s:%d, ", __FILE__, __LINE__); \
       printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
       exit(1); \
   } \
}

// High precision timer
double cpuSecond() {
   struct timespec ts;
   timespec_get(&ts, TIME_UTC);
   return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// CPU recursive reduction (Interleaved Pair)
int cpuRecursiveReduce(int *data, int const size) {
   if (size == 1) return data[0];
   int const stride = size / 2;
   for (int i = 0; i < stride; i++) {
       data[i] += data[i + stride];
   }
   return cpuRecursiveReduce(data, stride);
}

// GPU Neighbor Paired Approach
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int *idata = g_idata + blockIdx.x * blockDim.x;

   if (idx >= n) return;

   // In-place reduction in global memory
   for (int stride = 1; stride < blockDim.x; stride *= 2) {
       if ((tid % (2 * stride)) == 0) {
           idata[tid] += idata[tid + stride];
       }
       __syncthreads();
   }

   if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// GPU Recursive Reduction with sync
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize) {
   unsigned int tid = threadIdx.x;
   int *idata = g_idata + blockIdx.x * blockDim.x;
   int *odata = &g_odata[blockIdx.x];

   // Base case
   if (isize == 2 && tid == 0) {
       g_odata[blockIdx.x] = idata[0] + idata[1];
       return;
   }

   int istride = isize >> 1;
   if(istride > 1 && tid < istride) {
       idata[tid] += idata[tid + istride];
   }
   __syncthreads();

   if(tid == 0) {
       gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
       cudaDeviceSynchronize();
   }
   __syncthreads();
}

// GPU Recursive Reduction without sync
__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int isize) {
   unsigned int tid = threadIdx.x;
   int *idata = g_idata + blockIdx.x * blockDim.x;
   int *odata = &g_odata[blockIdx.x];

   if (isize == 2 && tid == 0) {
       g_odata[blockIdx.x] = idata[0] + idata[1];
       return;
   }

   int istride = isize >> 1;
   if(istride > 1 && tid < istride) {
       idata[tid] += idata[tid + istride];
       if(tid == 0) {
           gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
       }
   }
}

// GPU Recursive Reduction with fixed stride
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int iStride, int const iDim) {
   int *idata = g_idata + blockIdx.x * iDim;
   
   if (iStride == 1 && threadIdx.x == 0) {
       g_odata[blockIdx.x] = idata[0] + idata[1];
       return;
   }
   
   idata[threadIdx.x] += idata[threadIdx.x + iStride];
   
   if(threadIdx.x == 0 && blockIdx.x == 0) {
       gpuRecursiveReduce2<<<gridDim.x, iStride/2>>>(g_idata, g_odata, iStride/2, iDim);
   }
}

int main(int argc, char **argv) {
   // Device setup
   int dev = 0, gpu_sum;
   cudaDeviceProp deviceProp;
   CHECK(cudaGetDeviceProperties(&deviceProp, dev));
   printf("%s starting reduction at device %d: %s\n", argv[0], dev, deviceProp.name);
   CHECK(cudaSetDevice(dev));

   // Problem Size & Execution Configuration
   int size = 1 << 22;  // Total elements
   int blocksize = (argc > 1) ? atoi(argv[1]) : 512;  // Block size
   
   dim3 block(blocksize, 1);
   dim3 grid((size + block.x - 1) / block.x, 1);
   printf("Array size: %d, Grid: %d, Block: %d\n", size, grid.x, block.x);

   // Memory Allocation & Initialization
   size_t bytes = size * sizeof(int);
   int *h_idata = (int *)malloc(bytes);
   int *h_odata = (int *)malloc(grid.x * sizeof(int));
   int *tmp = (int *)malloc(bytes);

   for (int i = 0; i < size; i++) {
       h_idata[i] = 1;  // Set all elements to 1 for easy verification
   }
   memcpy(tmp, h_idata, bytes);

   int *d_idata = NULL;
   int *d_odata = NULL;
   CHECK(cudaMalloc((void **)&d_idata, bytes));
   CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

   // 1. CPU Recursive Reduction (baseline)
   double iStart = cpuSecond();
   int cpu_sum = cpuRecursiveReduce(tmp, size);
   double iElaps = cpuSecond() - iStart;
   printf("\n1. CPU Recursive:\t\t %.6f sec\tsum: %d\n", iElaps, cpu_sum);

   // 2. GPU Neighbor Paired
   CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
   iStart = cpuSecond();
   reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
   CHECK(cudaDeviceSynchronize());
   
   CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
   gpu_sum = 0;
   for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
   iElaps = cpuSecond() - iStart;
   printf("2. GPU Neighbor:\t\t %.6f sec\tsum: %d\n", iElaps, gpu_sum);

   // 3. GPU Recursive with sync
   CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
   iStart = cpuSecond();
   gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
   CHECK(cudaDeviceSynchronize());
   
   CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
   gpu_sum = 0;
   for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    iElaps = cpuSecond() - iStart;
   printf("3. GPU Recursive Sync:\t\t %.6f sec\tsum: %d\n", iElaps, gpu_sum);

   // 4. GPU Recursive without sync
   CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
   iStart = cpuSecond();
   gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, block.x);
   CHECK(cudaDeviceSynchronize());
   
   CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
   gpu_sum = 0;
   for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
   iElaps = cpuSecond() - iStart;
   printf("4. GPU Recursive NoSync:\t %.6f sec\tsum: %d\n", iElaps, gpu_sum);

   // 5. GPU Recursive with fixed stride
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    gpuRecursiveReduce2<<<grid, block.x/2>>>(d_idata, d_odata, block.x/2, block.x);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    iElaps = cpuSecond() - iStart;
    printf("5. GPU Recursive Fixed:\t %.6f sec\tsum: %d\n", iElaps, gpu_sum);

   // Cleanup
   free(h_idata);
   free(h_odata);
   free(tmp);
   CHECK(cudaFree(d_idata));
   CHECK(cudaFree(d_odata));
   CHECK(cudaDeviceReset());

   return 0;
}
