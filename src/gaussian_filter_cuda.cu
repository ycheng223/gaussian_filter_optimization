#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "gaussian_filter.h"
#include "gaussian_processing.h"


// Warmup GPU so 1st run isn't super slow
__global__ void warmup_kernel() {
      // Empty kernel that does minimal work
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx == 0) {
          // Minimal operation to ensure kernel actually runs
          int dummy = 1 + 1;
      }
  }

// C wrapper function to launch warmup kernel
extern "C" void warmup_gpu(void) {
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

// Device helper functions
__device__ void convolve_pixel_horizontal(
    unsigned char* image,
    int x, int y, int width, int height,
    const float* kernel, int kernel_size,
    float* out_r, float* out_g, float* out_b
) {
    int range = kernel_size / 2;
    *out_r = *out_g = *out_b = 0.0f;

    for (int k = -range; k <= range; ++k) {
        int neighbor_x = x + k;
        if (neighbor_x >= 0 && neighbor_x < width) {
            int base_idx = (y * width + neighbor_x) * 4;
            float weight = kernel[k+range];

            *out_r += image[base_idx + 0] * weight;
            *out_g += image[base_idx + 1] * weight;
            *out_b += image[base_idx + 2] * weight;
        }
    }
}

// Vertical convolution helper
__device__ void convolve_pixel_vertical(
    unsigned char* image,
    int x, int y, int width, int height,
    const float* kernel, int kernel_size,
    float* out_r, float* out_g, float* out_b
) {
    int range = kernel_size / 2;
    *out_r = *out_g = *out_b = 0.0f;

    for (int k = -range; k <= range; ++k) {
        int neighbor_y = y + k;
        if (neighbor_y >= 0 && neighbor_y < height) {
            int base_idx = (neighbor_y * width + x) * 4;
            float weight = kernel[k + range];

            *out_r += image[base_idx + 0] * weight;
            *out_g += image[base_idx + 1] * weight;
            *out_b += image[base_idx + 2] * weight;
        }
    }
}

// This is the kernel (runs on all SMs)
__global__ void gaussian_filter_cuda_convolve(
    unsigned char* dev_in, unsigned char* dev_out,
    int width, int height, const float* kernel, int kernel_size, int direction // 0 = horizontal, 1 = vertical
) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // returns global index position (i.e. block position + thread position)
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            float r, g, b;

            if (direction == 0){
                convolve_pixel_horizontal(dev_in, x, y, width, height, kernel, kernel_size, &r, &g, &b);
            } else {
                convolve_pixel_vertical(dev_in, x, y, width, height, kernel, kernel_size, &r, &g, &b);
            }

            int out_idx = (y * width + x) * 4;
            dev_out[out_idx + 0] = (uint8_t)r;
            dev_out[out_idx + 1] = (uint8_t)g;
            dev_out[out_idx + 2] = (uint8_t)b;
            dev_out[out_idx + 3] = dev_in[out_idx + 3]; // copy the alpha channel directly from input to output buffer
        }
    }


// Entry point
extern "C" __host__ void gaussian_filter_cuda(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    
    // Validate input parameters
    if (!image) {
        fprintf(stderr, "CUDA: NULL image pointer\n");
        return;
    }
    if (width <= 0 || height <= 0 || kernel_size <= 0) {
        fprintf(stderr, "CUDA: Invalid dimensions w=%d h=%d k=%d\n", width, height, kernel_size);
        return;
    }
    

    // Precompute the 1D Gaussian kernel
    float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
    if (!kernel) return;

    // Declare dimensions
    size_t image_size = width * height * CHANNELS_PER_PIXEL * sizeof(unsigned char);
    size_t kernel_size_bytes = kernel_size * sizeof(float);

    dim3 blockSize(16, 16); // Create 2D block of size 16x16 -> 256 threads (8 warps) -> easily fits in 4070
    dim3 gridSize((width + 15) / 16, (height + 15) / 16); // Calculates grid size (i.e. number of blocks to cover image)
    // +15 ensures coverage of partial blocks

    // Allocate GPU memory
    unsigned char *dev_in, *dev_temp, *dev_out;
    float *dev_kernel;

    cudaError_t err;
    err = cudaMalloc((void**)&dev_in, image_size); 
    // double pointer because we are modifying the pointer itself (write a GPU memory address into pointer variable), not it's data
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_in failed: %s\n", cudaGetErrorString(err));
        free(kernel); return;
    }

    err = cudaMalloc((void**)&dev_temp, image_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_temp failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_in); free(kernel); return;
    }

    err = cudaMalloc((void**)&dev_out, image_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_out failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_in); cudaFree(dev_temp); free(kernel); return;
    }

    err = cudaMalloc((void**)&dev_kernel, kernel_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_in); cudaFree(dev_temp); cudaFree(dev_out); free(kernel); return;
    }


    // Transfer original image from system memory to VRAM
      err = cudaMemcpy(dev_in, image, image_size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr, "cudaMemcpy H2D image failed: %s\n", cudaGetErrorString(err));
          goto cleanup;
      }

      err = cudaMemcpy(dev_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr, "cudaMemcpy H2D kernel failed: %s\n", cudaGetErrorString(err));
          goto cleanup;
      }

    // LAUNCH THE KERNELS

    // Horizontal CUDA Pass
    gaussian_filter_cuda_convolve<<<gridSize, blockSize>>>(
        dev_in, dev_temp, width, height, dev_kernel, kernel_size, 0  // Original width/height
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Horizontal kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Vertical CUDA Pass
    gaussian_filter_cuda_convolve<<<gridSize, blockSize>>>(
        dev_temp, dev_out, width, height, dev_kernel, kernel_size, 1
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Vertical kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Wait for kernels to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel synchronization failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy convolved result from VRAM back to system memory
    err = cudaMemcpy(image, dev_out, image_size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H (final) failed: %s\n", cudaGetErrorString(err));
    }

    cleanup:
        cudaFree(dev_in);
        cudaFree(dev_temp);
        cudaFree(dev_out);
        cudaFree(dev_kernel);
        free(kernel);
}