#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "gaussian_filter.h"
#include "gaussian_processing.h"
#include "gaussian_filter_cuda.h"

// Constant memory declaration (must be global scope)
#define MAX_KERNEL_SIZE 128
__constant__ float const_kernel[MAX_KERNEL_SIZE];

__device__ void convolve_pixel_horizontal_constant( unsigned char* image, int x, int y, int width, int height, 
                                            int kernel_size, float* out_r, float* out_g, float* out_b) {

    int range = kernel_size / 2;
    *out_r = *out_g = *out_b = 0.0f;
    
    for (int k = -range; k <= range; ++k) {
        int neighbor_x = x + k; // horizontal movement
        if (neighbor_x >= 0 && neighbor_x < width) {
            int base_idx = (y * width + neighbor_x) * 4;
            float weight = const_kernel[k + range]; // read from constant memory

            *out_r += image[base_idx + 0] * weight;
            *out_g += image[base_idx + 1] * weight;
            *out_b += image[base_idx + 2] * weight;
        }
    }
}

__device__ void convolve_pixel_vertical_constant( unsigned char* image, int x, int y, int width, int height, 
                                            int kernel_size, float* out_r, float* out_g, float* out_b) {

    int range = kernel_size / 2;
    *out_r = *out_g = *out_b = 0.0f;
    
    for (int k = -range; k <= range; ++k) {
        int neighbor_y = y + k; // vertical movement
        if (neighbor_y >= 0 && neighbor_y < height) {
            int base_idx = (neighbor_y * width + x) * 4;
            float weight = const_kernel[k + range]; // read from constant memory

            *out_r += image[base_idx + 0] * weight;
            *out_g += image[base_idx + 1] * weight;
            *out_b += image[base_idx + 2] * weight;
        }
    }
}

// This is the kernel (runs on all SMs)
// direction: 0 = horizontal, 1 = vertical
__global__ void gaussian_filter_cuda_convolve_constant( unsigned char* dev_in, unsigned char* dev_out, int width, int height, 
                                                        int kernel_size, int direction)  {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // returns global index position (i.e. block position + thread position)
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r, g, b;

        if (direction == 0){
            convolve_pixel_horizontal_constant(dev_in, x, y, width, height, kernel_size, &r, &g, &b);
        } else {
            convolve_pixel_vertical_constant(dev_in, x, y, width, height, kernel_size, &r, &g, &b);
        }

        int out_idx = (y * width + x) * 4;
        dev_out[out_idx + 0] = (uint8_t)r;
        dev_out[out_idx + 1] = (uint8_t)g;
        dev_out[out_idx + 2] = (uint8_t)b;
        dev_out[out_idx + 3] = dev_in[out_idx + 3]; // copy the alpha channel directly from input to output buffer
    }
}

// Helper function to allocate GPU memory
cudaError_t allocate_device_memory(unsigned char** dev_in, unsigned char** dev_temp, unsigned char** dev_out, // double pointer because we are modifying caller's pointer
                                    size_t image_size) {
    cudaError_t err;

    // Allocate dev_in
    err = cudaMalloc((void**)dev_in, image_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_in failed: %s\n", cudaGetErrorString(err));
        return err;  // Return error, let caller handle cleanup
    }

    // Allocate dev_temp
    err = cudaMalloc((void**)dev_temp, image_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_temp failed: %s\n", cudaGetErrorString(err));
        return err;  // Caller will clean up dev_in
    }

    // Allocate dev_out
    err = cudaMalloc((void**)dev_out, image_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_out failed: %s\n", cudaGetErrorString(err));
        return err;  // Caller will clean up dev_in and dev_temp
    }

    return cudaSuccess;  // All allocations succeeded
}

// Helper function to copy image from host (CPU) to device (GPU) memory
cudaError_t copy_to_device(unsigned char* dev_in, unsigned char* image, float* kernel, size_t image_size, int kernel_size) {
    cudaError_t err;

    // Copy image to device
    err = cudaMemcpy(dev_in, image, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D image failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Copy kernel to constant memory
    err = cudaMemcpyToSymbol(
        const_kernel,                    // Destination: __constant__ symbol
        kernel,                          // Source: host memory
        kernel_size * sizeof(float),
        0,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

// Helper fn to launch kernels
cudaError_t launch_convolution_kernels(unsigned char* dev_in, unsigned char* dev_temp, unsigned char* dev_out,
                                        int width, int height, int kernel_size, dim3 gridSize, dim3 blockSize) {
    cudaError_t err;

    // Horizontal pass
    gaussian_filter_cuda_convolve_constant<<<gridSize, blockSize>>>(
        dev_in, dev_temp, width, height, kernel_size, 0  // direction=0 (horizontal)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Horizontal kernel launch failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Vertical pass
    gaussian_filter_cuda_convolve_constant<<<gridSize, blockSize>>>(
        dev_temp, dev_out, width, height, kernel_size, 1  // direction=1 (vertical)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Vertical kernel launch failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel synchronization failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

// Helper to copy image back to host
cudaError_t copy_to_host(unsigned char* host_image, unsigned char* dev_out, size_t image_size) {
    cudaError_t err;

    err = cudaMemcpy(host_image, dev_out, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}


// Host function
  extern "C" __host__ void gaussian_filter_cuda_constant(unsigned char* image, int width, int height, float sigma, int kernel_size) {
      // Validate inputs
      if (!image) {
          fprintf(stderr, "CUDA: NULL image pointer\n");
          return;
      }
      if (width <= 0 || height <= 0 || kernel_size <= 0) {
          fprintf(stderr, "CUDA: Invalid dimensions w=%d h=%d k=%d\n", width, height, kernel_size);
          return;
      }

      // Precompute Gaussian kernel on CPU
      float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
      if (!kernel) return;

      // Calculate sizes
      size_t image_size = width * height * CHANNELS_PER_PIXEL * sizeof(unsigned char);

      // Configure launch parameters
      dim3 blockSize(16, 16);
      dim3 gridSize((width + 15) / 16, (height + 15) / 16);

      // Initialize device pointers to NULL (important!)
      // cudaFree(NULL) is safe and does nothing
      unsigned char *dev_in = NULL;
      unsigned char *dev_temp = NULL;
      unsigned char *dev_out = NULL;

      cudaError_t err;

      // Allocate device memory
      err = allocate_device_memory(&dev_in, &dev_temp, &dev_out, image_size);
      if (err != cudaSuccess) {
          goto cleanup;  // Jump to cleanup (same function, so this works!)
      }
      // Copy image + kernel data to device
      err = copy_to_device(dev_in, image, kernel, image_size, kernel_size);
      if (err != cudaSuccess) {
          goto cleanup;
      }
      // Launch convolutional kernels
      err = launch_convolution_kernels(dev_in, dev_temp, dev_out, width, height, kernel_size, gridSize, blockSize);
      if (err != cudaSuccess) {
          goto cleanup;
      }
      // Copy result back to host
      err = copy_to_host(image, dev_out, image_size);
      if (err != cudaSuccess) {
          goto cleanup;
      }

  cleanup:
      // Free device memory
      if (dev_in)   cudaFree(dev_in);
      if (dev_temp) cudaFree(dev_temp);
      if (dev_out)  cudaFree(dev_out);

      // Free host memory
      free(kernel);
  }