#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <time.h>

#include <complex.h>
#include <math.h>

#include <cuComplex.h>



// This is used to return and propagate an EXIT_FAILURE return value. 
#define CHECK_RET(ret) if (ret == EXIT_FAILURE) { return EXIT_FAILURE; }

// In case of an erroneous condition, this macro prints the current file and 
// line of an error, a useful error message, and then returns EXIT_FAILURE. 
#define CHECK(condition, err_fmt, err_msg) \
    if (condition) { \
        printf(err_fmt " (%s:%d)\n", err_msg, __FILE__, __LINE__); \
        return EXIT_FAILURE; \
    }

// This macro insures a call to malloc succeeded.
#define CHECK_MALLOC(p, name) \
    CHECK(!(p), "Failed to allocate %s", name)

// This macro insures a call to a CUDA function succeeded.
#define CHECK_CUDA(stat) \
    CHECK((stat) != cudaSuccess, "CUDA error %s", cudaGetErrorString(stat))


// This function reverses a 32-bit bitstring.
uint32_t reverse_bits(uint32_t x) {
    // 1. Swap the position of consecutive bits
    // 2. Swap the position of consecutive pairs of bits
    // 3. Swap the position of consecutive quads of bits
    // 4. Continue this until swapping the two consecutive 16-bit parts of x
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

int fft(const float complex* x, float complex* Y, uint32_t N) {
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    for (uint32_t i = 0; i < N; i++) {
        // Reverse the 32-bit index.
        uint32_t rev = reverse_bits(i);

        // Only keep the last logN bits of the output.
        rev = rev >> (32 - logN);

        // Base case: set the output to the bit-reversed input.
        Y[i] = x[rev];
    }

    // Set m to 2, 4, 8, 16, ..., N
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int mh = 1 << (s - 1);

        float complex twiddle = cexpf(-2.0I * M_PI / m);

        // Iterate through Y in strides of length m=2**s
        // Set k to 0, m, 2m, 3m, ..., N-m
        for (uint32_t k = 0; k < N; k += m) {
            float complex twiddle_factor = 1;

            // Set both halves of the Y array at the same time
            // j = 1, 4, 8, 16, ..., N / 2
            for (int j = 0; j < mh; j++) {
                float complex a = Y[k + j];
                float complex b = twiddle_factor * Y[k + j + mh];

                // Compute pow(twiddle, j)
                twiddle_factor *= twiddle;

                Y[k + j] = a + b;
                Y[k + j + mh] = a - b;
            }
        }

    }
    return EXIT_SUCCESS;
}


// This is used by `fft_gpu`.
// This FFT algorithm works just like the Cooley-Tukey algorithm,
// except a single thread is in charge of each of the N elements.
// Threads synchronize in order to traverse the array log N times.
__global__ void fft_kernel(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N, int logN) {
    // Find this thread's index in the input array.
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // Start by bit-reversing the input.
    // Reverse the 32-bit index.
    // Only keep the last logN bits of the output.
    uint32_t rev;

    rev = reverse_bits_gpu(2 * i);
    rev = rev >> (32 - logN);
    Y[2 * i] = x[rev];

    rev = reverse_bits_gpu(2 * i + 1);
    rev = rev >> (32 - logN);
    Y[2 * i + 1] = x[rev];

    __syncthreads();

    // Set mh to 1, 2, 4, 8, ..., N/2
    for (int s = 1; s <= logN; s++) {
        int mh = 1 << (s - 1);  // 2 ** (s - 1)

        // k = 2**s * (2*i // 2**(s-1))  for i=0..N/2-1
        // j = i % (2**(s - 1))  for i=0..N/2-1
        int k = threadIdx.x / mh * (1 << s);
        int j = threadIdx.x % mh;
        int kj = k + j;

        cuFloatComplex a = Y[kj];

        // exp(-2i pi j / 2**s)
        // exp(-2i pi j / m)
        // exp(-i pi j / (m/2))
        // exp(ix)
        // cos(x) + i sin(x)
        float tr;
        float ti;

        // TODO possible optimization:
        // pre-compute twiddle factor array
        // twiddle[s][j] = exp(-i pi * j / 2**(s-1))
        // for j=0..N/2-1 (proportional)
        // for s=1..log N
        // need N log N / 2 tmp storage...

        // Compute the sine and cosine to find this thread's twiddle factor.
        sincosf(-(float)M_PI * j / mh, &ti, &tr);
        cuFloatComplex twiddle = make_cuFloatComplex(tr, ti);

        cuFloatComplex b = cuCmulf(twiddle, Y[kj + mh]);

        // Set both halves of the Y array at the same time
        Y[kj] = cuCaddf(a, b);
        Y[kj + mh] = cuCsubf(a, b);

        // Wait for all threads to finish before traversing the array once more.
        __syncthreads();
    }
}

int fft_gpu(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N) {
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    cudaError_t st;

    // Allocate memory on the CUDA device.
    cuFloatComplex* x_dev;
    cuFloatComplex* Y_dev;
    st = cudaMalloc((void**)&Y_dev, sizeof(*Y) * N);
    // Check for any CUDA errors
    CHECK_CUDA(st);

    st = cudaMalloc((void**)&x_dev, sizeof(*x) * N);
    CHECK_CUDA(st);

    // Copy input array to the device.
    st = cudaMemcpy(x_dev, x, sizeof(*x) * N, cudaMemcpyHostToDevice);
    CHECK_CUDA(st);

    // Send as many threads as possible per block.
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, cuda_device_ix);
    CHECK_CUDA(st);

    // Create one thread for every two elements in the array 
    int size = N >> 1;
    int block_size = min(size, prop.maxThreadsPerBlock);
    dim3 block(block_size, 1);
    dim3 grid((size + block_size - 1) / block_size, 1);

    // Call the kernel
    fft_kernel <<< grid, block >>> (x_dev, Y_dev, N, logN);

    // Copy the output
    st = cudaMemcpy(Y, Y_dev, sizeof(*x) * N, cudaMemcpyDeviceToHost);
    CHECK_CUDA(st);

    // Free CUDA memory
    st = cudaFree(x_dev);
    CHECK_CUDA(st);
    st = cudaFree(Y_dev);
    CHECK_CUDA(st);

    return EXIT_SUCCESS;
}





int main(int argc, const char** argv)
{
    // Default value of N here. Should be a power of two if using the fft algorithm. ///
    uint32_t N = 8;

    // Program options
    int no_sample = false;
    int measure_time = false;
    int fill_with = 0;
    bool no_print = false;
    char* algorithm = NULL; 

    static const char *const usage[] = {
        "fft [options]",
        NULL,
    };

    // Setup program argument parser
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Algorithm and data options"),
        OPT_STRING('a', "algorithm", &algorithm, "algorithm for computing the DFT (dft|fft|gpu|fft_gpu|dft_gpu), default is 'dft'"),
        OPT_INTEGER('f', "fill_with", &fill_with, "fill data with this integer"),
        OPT_BOOLEAN('s', "no_samples", &no_sample, "do not set first part of array to sample data"),
        OPT_INTEGER('N', "data_length", &N, "data length"),
        OPT_GROUP("Benchmark options"),
        OPT_INTEGER('t', "measure_time", &measure_time, "measure runtime. runs algorithms <int> times. set to 0 if not needed."),
        OPT_BOOLEAN('p', "no_print", &no_print, "do not print results"),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, 
        "\nCompute the FFT of a dataset with a given size, using a specified DFT algorithm.",
        "");
    argc = argparse_parse(&argparse, argc, argv);

    float complex* in;
    float complex* out;

    cuFloatComplex* in_gpu;
    cuFloatComplex* out_gpu;

    bool gpu = true;

    prof_info times[measure_time];

    // Check if a string is equal to the requested algorithm
    #define ALG_IS(s) (strcmp(algorithm, s) == 0)

    // Setup and run the given algorithm
    // Setup is different for GPU and CPU algorithms
    if (algorithm == NULL || ALG_IS("dft")) {
        gpu = false;
        // CHECK_RET causes the program to exit in case of any error 
        // For example, if setup_data returns EXIT_FAILURE, main should return EXIT_FAILURE
        CHECK_RET(setup_data(&in, &out, N, fill_with, no_sample));
        CHECK_RET(run("O(N^2) DFT Algorithm", (algorithm_t)dft, in, out, N, measure_time, times));
    } else if (ALG_IS("fft")) {
        gpu = false;
        CHECK_RET(setup_data(&in, &out, N, fill_with, no_sample));
        CHECK_RET(run("Cooley-Tukey FFT", (algorithm_t)fft, in, out, N, measure_time, times));
    } else if (ALG_IS("fft_gpu") || ALG_IS("gpu")) {
        CHECK_RET(setup_gpu(&in_gpu, &out_gpu, N, fill_with, no_sample));
        CHECK_RET(run("Cooley-Tukey FFT on GPU", (algorithm_t)fft_gpu, in_gpu, out_gpu, N, measure_time, times));
    } else if (ALG_IS("dft_gpu")) {
        CHECK_RET(setup_gpu(&in_gpu, &out_gpu, N, fill_with, no_sample));
        CHECK_RET(run("O(N^2) DFT on GPU", (algorithm_t)dft_gpu, in_gpu, out_gpu, N, measure_time, times));
    } else {
        printf("Algorithm '%s' unknown\n", algorithm);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < measure_time; i++) {
        printf("%14.8f (s)\n", times[i].seconds);
    }
    // Print the results
    if (!no_print) {
        if (gpu) {
            show_complex_gpu_vector(out_gpu, N);
        } else {
            show_complex_vector(out, N);
        }
    }
    if (!gpu) {
        free(in);
        free(out);
    } else {
        free(in_gpu);
        free(out_gpu);
    }
    return 0;
}
