#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Error checking macro
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Add these CUDA-specific utility functions
__device__ inline unsigned char clamp(int value) {
    return (value < 0) ? 0 : ((value > 255) ? 255 : value);
}

// Kernel for RGB to Grayscale conversion
__global__ void rgbToGrayscaleKernel(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int grayIdx = y * width + x;
        d_output[grayIdx] = (unsigned char)(0.299f * d_input[idx] + 0.587f * d_input[idx+1] + 0.114f * d_input[idx+2]);
    }
}


// Kernel for image flipping
__global__ void flipKernel(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, bool isHorizontal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int outputIdx;
        int inputIdx = (y * width + x) * channels;

        if (isHorizontal) {
            outputIdx = (y * width + (width - 1 - x)) * channels;
        } else {
            outputIdx = ((height - 1 - y) * width + x) * channels;
        }

        for (int c = 0; c < channels; ++c) {
            d_output[outputIdx + c] = d_input[inputIdx + c];
        }
    }
}

// Kernel for image blurring (not optimal)
__global__ void blurKernel(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, int blurRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0;
            int count = 0;
            
            for (int dy = -blurRadius; dy <= blurRadius; dy++) {
                for (int dx = -blurRadius; dx <= blurRadius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += d_input[(ny * width + nx) * channels + c];
                        count++;
                    }
                }
            }
            
            d_output[(y * width + x) * channels + c] = (unsigned char)(sum / count);
        }
    }
}

// Kernel for 2D convolution (not optimal)
__global__ void convolution2DKernel(unsigned char* d_input, unsigned char* d_output, float* d_filter, 
                                    int width, int height, int channels, int filterSize, float filterSum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float result = 0.0f;
            for (int fy = 0; fy < filterSize; fy++) {
                for (int fx = 0; fx < filterSize; fx++) {
                    int imageX = x + fx - radius;
                    int imageY = y + fy - radius;
                    
                    if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                        float pixelValue = d_input[(imageY * width + imageX) * channels + c];
                        float filterValue = d_filter[fy * filterSize + fx];
                        result += pixelValue * filterValue;
                    }
                }
            }
            d_output[(y * width + x) * channels + c] = clamp(int(result));
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <image_file> <operation> <block_size_x> <block_size_y> [additional_params]\n", argv[0]);
        printf("Operations: grayscale, blur, flip, convolution\n");
        return 1;
    }

    // Parse command line arguments
    const char* inputFile = argv[1];
    const char* operation = argv[2];
    int blockSize_x = atoi(argv[3]);
    int blockSize_y = atoi(argv[4]);

    // Load the image
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // Allocate device memory
    int imageSize = width * height * channels;
    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc((void **)&d_input, imageSize));
    CHECK(cudaMalloc((void **)&d_output, imageSize));
    CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 block(blockSize_x, blockSize_y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Perform the requested operation
    if (strcmp(operation, "grayscale") == 0) {
        rgbToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height);
        channels = 1; // Output is grayscale
    } else if (strcmp(operation, "blur") == 0) {
        int blurRadius = (argc > 4) ? atoi(argv[4]) : 1;
        blurKernel<<<grid, block>>>(d_input, d_output, width, height, channels, blurRadius);
    } else if (strcmp(operation, "flip") == 0) {
        bool isHorizontal = (argc > 4) ? (strcmp(argv[4], "horizontal") == 0) : true;
        flipKernel<<<grid, block>>>(d_input, d_output, width, height, channels, isHorizontal);
    } else if (strcmp(operation, "convolution") == 0) {
        if (argc < 6) {
          printf("Usage: %s <image_file> convolution <block_size> <filter_type> <direction>\n", argv[0]);
          printf("Filter types: sobel\n");
          printf("Directions for Sobel: horizontal, vertical\n");
          return 1;
      }
      
      int filterSize = 3; // Sobel filter is always 3x3
      float* h_filter = (float*)malloc(filterSize * filterSize * sizeof(float));
      
      const char* filterType = argv[5];
      const char* direction = argv[6];

      if (strcmp(filterType, "sobel") == 0) {
          // Define Sobel filters
          float sobel_horizontal[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
          float sobel_vertical[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

          // Choose the appropriate filter based on the direction
          if (strcmp(direction, "horizontal") == 0) {
              memcpy(h_filter, sobel_horizontal, filterSize * filterSize * sizeof(float));
          } else if (strcmp(direction, "vertical") == 0) {
              memcpy(h_filter, sobel_vertical, filterSize * filterSize * sizeof(float));
          } else {
              printf("Invalid direction for Sobel filter. Use 'horizontal' or 'vertical'.\n");
              return 1;
          }
      } else {
          printf("Unsupported filter type. Currently only 'sobel' is supported.\n");
          return 1;
      }
      
      // No normalization
      float filterSum = 1.0f;
      
      float* d_filter;
      CHECK(cudaMalloc((void **)&d_filter, filterSize * filterSize * sizeof(float)));
      CHECK(cudaMemcpy(d_filter, h_filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice));
      
      // Launch the convolution kernel
      convolution2DKernel<<<grid, block>>>(d_input, d_output, d_filter, width, height, channels, filterSize, filterSum);
      
      CHECK(cudaFree(d_filter));
      free(h_filter);
    } else {
        printf("Unknown operation: %s\n", operation);
        return 1;
    }

    // Check for errors
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));

    // Save the output image
    char outputFilename[256];
    snprintf(outputFilename, sizeof(outputFilename), "output_%s.png", operation);
    stbi_write_png(outputFilename, width, height, channels, h_output, width * channels);
    printf("Output saved as %s\n", outputFilename);

    // Clean up
    stbi_image_free(h_input);
    free(h_output);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaDeviceReset());

    return 0;
}



