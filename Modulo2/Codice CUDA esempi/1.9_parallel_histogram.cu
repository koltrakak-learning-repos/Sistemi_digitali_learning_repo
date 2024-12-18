#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>

#define NUM_BINS 7          // 26 letters divided by 4, rounded up
#define BLOCK_SIZE 128
#define MAX_FILE_SIZE (1 << 26) 

// Error checking macro
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Sequential CPU version
void hist_sequential(const char *data, int length, unsigned int *histo) {
    for (int i = 0; i < length; i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position / 4]++;
        }
    }
}

// Results verification
bool verify_results(const unsigned int *hist1, const unsigned int *hist2) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (hist1[i] != hist2[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n", i, hist1[i], hist2[i]);
            return false;
        }
    }
    return true;
}

// Print histogram results
void print_histogram(const unsigned int *hist) {
    printf("\nHistogram results:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d (letters %c-%c): %u\n",
               i,
               'a' + (i * 4),
               'a' + (i * 4 + 3 > 25 ? 25 : i * 4 + 3),
               hist[i]);
    }
    printf("\n");
}

__global__ void hist_block_per_elememt(unsigned char *buffer, long size, unsigned int *histo) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {  // Check if thread id is within bounds
        int alphabet_position = buffer[tid] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position/4]), 1);
        }
    }
}

__global__ void hist_block_per_elememt_SMEM(unsigned char* input, unsigned int* bins,
                                                unsigned int num_elements, unsigned int num_bins) {
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
   extern __shared__ unsigned int histo_s[];

   // Initialization of shared memory
   for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
       histo_s[binIdx] = 0u;
   }
   __syncthreads();
   
   // Each thread processes only one element
   if(tid < num_elements) {
       int alphabet_position = input[tid] - 'a';
       if (alphabet_position >= 0 && alphabet_position < 26) {
           atomicAdd(&(histo_s[alphabet_position/4]), 1);
       }
   }
   __syncthreads();
   
   // Global histogram update
   for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
       atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
   }
}

__global__ void hist_block_partitioned(unsigned char *buffer, long size, unsigned int *histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int section_size = (size-1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section_size;
    
    for (int k = 0; k < section_size; k++) {
        if (start + k < size) {
            int alphabet_position = buffer[start+k] - 'a';
            if (alphabet_position >= 0 && alphabet_position < 26) {
                atomicAdd(&(histo[alphabet_position/4]), 1);
            }
        }
    }
}

__global__ void hist_block_partitioned_SMEM(unsigned char* input, unsigned int* bins,
                                          unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ unsigned int histo_s[];
    
    // Initialize shared memory histogram
    for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    __syncthreads();
    
    // Calculate section size for block partitioning
    int section_size = (num_elements - 1) / (blockDim.x * gridDim.x) + 1;
    int start = tid * section_size;
    
    // Process section
    for(int k = 0; k < section_size; k++) {
        if(start + k < num_elements) {
            int alphabet_position = input[start + k] - 'a';
            if(alphabet_position >= 0 && alphabet_position < 26) {
                atomicAdd(&(histo_s[alphabet_position/4]), 1);
            }
        }
    }
    __syncthreads();
    
    // Update global histogram
   for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
       atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
   }
}

template<int UNROLL_FACTOR>
__global__ void hist_block_partitioned_SMEM_unrolled(unsigned char* input, unsigned int* bins,
                                                    unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ unsigned int histo_s[];
    
    // Init
    if (threadIdx.x < 7) { // Considerando che il numero di bin è noto e piccolo
        histo_s[threadIdx.x] = 0u;
    }
    __syncthreads();

    // Compute section size
    int section_size = (num_elements - 1) / (blockDim.x * gridDim.x) + 1;
    int start = tid * section_size;

    // Unrolled section processing
    if (start + (UNROLL_FACTOR-1) < num_elements) {
        #pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
            unsigned char c = input[start + k];
            if ((unsigned)(c - 'a') < 26u) {
                atomicAdd(&(histo_s[(c - 'a') >> 2]), 1);
            }
        }
    }

    // Handle remaining elements
    for (int k = UNROLL_FACTOR; k < section_size && start + k < num_elements; k++) {
        unsigned char c = input[start + k];
        if ((unsigned)(c - 'a') < 26u) {
            atomicAdd(&(histo_s[(c - 'a') >> 2]), 1);
        }
    }
    __syncthreads();

    // Final update
    if (threadIdx.x < 7) {
        atomicAdd(&(bins[threadIdx.x]), histo_s[threadIdx.x]);
    }
}

__global__ void hist_interleaved(unsigned char *buffer, long size, unsigned int *histo) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int alphabet_position = buffer[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position/4]), 1);
        }
    }
}

__global__ void hist_interleaved_SMEM(unsigned char* input, unsigned int* bins,
                                          unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int histo_s[];
    
    for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        histo_s[binIdx] = 0u;
    }

    __syncthreads();

    for(unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
        int alphabet_position = input[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    __syncthreads();
    
    for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
    }
}

template<int UNROLL_FACTOR>
__global__ void hist_interleaved_SMEM_unrolled(unsigned char* input, unsigned int* bins,
                                              unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int histo_s[];
    
    // Init
    if (threadIdx.x < 7) { // Considerando che il numero di bin è noto e piccolo
        histo_s[threadIdx.x] = 0u;
    }
    __syncthreads();
    
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int base = tid;
    
    // Unrolled computation
    if (base + (UNROLL_FACTOR-1) * stride < num_elements) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; j++) {
            unsigned char c = input[base + j * stride];
            if ((unsigned)(c - 'a') < 26u) {
                atomicAdd(&(histo_s[(c - 'a') >> 2]), 1);
            }
        }
    }
    
    // Remaining elements
    for(unsigned int i = base + UNROLL_FACTOR * stride; i < num_elements; i += stride) {
        unsigned char c = input[i];
        if ((unsigned)(c - 'a') < 26u) {
            atomicAdd(&(histo_s[(c - 'a') >> 2]), 1);
        }
    }
    __syncthreads();
    
    // Unrolled final update
    if (threadIdx.x < 7) {
        atomicAdd(&(bins[threadIdx.x]), histo_s[threadIdx.x]);
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    // Configurable parameter for grid division
    constexpr int GRID_DIVISION_FACTOR = 4;  // It will result in 2^4 = 16
    
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // File reading
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) {
        printf("Could not open file %s\n", argv[1]);
        return 1;
    }
    
    unsigned char* h_data = (unsigned char*)malloc(MAX_FILE_SIZE);
    size_t bytes_read = fread(h_data, 1, MAX_FILE_SIZE, fp);
    fclose(fp);
    printf("Read %zu bytes from file\n", bytes_read);
    
    // Host memory allocation
    unsigned int* h_hist_cpu = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int* h_hist_gpu = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    
    // Device memory allocation
    unsigned char* d_data;
    unsigned int* d_hist;
    CHECK(cudaMalloc(&d_data, bytes_read));
    CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_data, h_data, bytes_read, cudaMemcpyHostToDevice));
    
    // Calculate number of blocks
    int num_blocks = (bytes_read + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks = num_blocks / (1 << GRID_DIVISION_FACTOR);
    
    clock_t start, end;
    double cpu_time, gpu_time;

    // CPU Version
    printf("\nRunning CPU version...\n");
    start = clock();
    hist_sequential((const char*)h_data, bytes_read, h_hist_cpu);
    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %.2f ms\n", cpu_time);
    print_histogram(h_hist_cpu);

    printf("\nRunning GPU versions...\n");

    // Kernel 0: One-to-one
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_block_per_elememt<<<num_blocks, BLOCK_SIZE>>>(d_data, bytes_read, d_hist);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 0 (one-to-one) time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 1: Block partitioned
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_block_partitioned<<<blocks, BLOCK_SIZE>>>(d_data, bytes_read, d_hist);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 1 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 2: Interleaved
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_interleaved<<<blocks, BLOCK_SIZE>>>(d_data, bytes_read, d_hist);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 2 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 3: One-to-one with shared memory
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_block_per_elememt_SMEM<<<num_blocks, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int)>>>
        (d_data, d_hist, bytes_read, NUM_BINS);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 3 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 4: Block partitioned with shared memory
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_block_partitioned_SMEM<<<blocks, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int)>>>
        (d_data, d_hist, bytes_read, NUM_BINS);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 4 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 5: Interleaved with shared memory
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_interleaved_SMEM<<<blocks, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int)>>>
        (d_data, d_hist, bytes_read, NUM_BINS);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 5 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 6: Block partitioned with shared memory and unrolling
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_block_partitioned_SMEM_unrolled<1 << GRID_DIVISION_FACTOR><<<blocks, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int)>>>
        (d_data, d_hist, bytes_read, NUM_BINS);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 6 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Kernel 7: Interleaved with shared memory and unrolling
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    start = clock();
    hist_interleaved_SMEM_unrolled<1 << GRID_DIVISION_FACTOR><<<blocks, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int)>>>
        (d_data, d_hist, bytes_read, NUM_BINS);
    CHECK(cudaDeviceSynchronize());
    end = clock();
    gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel 7 time: %.2f ms\n", gpu_time);
    verify_results(h_hist_cpu, h_hist_gpu);
    print_histogram(h_hist_gpu);

    // Cleanup
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_hist));
    free(h_data);
    free(h_hist_cpu);
    free(h_hist_gpu);
    
    return 0;
}
