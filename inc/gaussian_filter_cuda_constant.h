  #ifndef GAUSSIAN_FILTER_CUDA_CONSTANT_H
  #define GAUSSIAN_FILTER_CUDA_CONSTANT_H


  // Forward declare cuda type definitions if we are compiling with GCC (won't recognize these types otherwise)
  #ifdef __cplusplus
  #include <cuda_runtime.h>  // For cudaError_t and dim3
  #else
  typedef int cudaError_t;
  typedef struct {
      unsigned int x, y, z;
  } dim3;
  #endif

  #include "common.h"

  #ifndef __NVCC__
  #define __global__
  #define __host__
  #define __device__
  #define __constant__
  #endif

  #define MAX_KERNEL_SIZE 64


__device__ void convolve_pixel_horizontal_constant(unsigned char* image, int x, int y, int width, int height, 
                                                        int kernel_size, float* out_r, float* out_g, float* out_b);

__device__ void convolve_pixel_vertical_constant(unsigned char* image, int x, int y, int width, int height, 
                                                    int kernel_size, float* out_r, float* out_g, float* out_b);


__global__ void gaussian_filter_cuda_convolve_constant(unsigned char* dev_in, unsigned char* dev_out, int width, int height, 
                                                            int kernel_size, int direction);
void warmup_gpu(void);

  #ifdef __cplusplus
  extern "C" {
  #endif
  cudaError_t allocate_device_memory(unsigned char** dev_in, unsigned char** dev_temp, 
                                        unsigned char** dev_out, size_t image_size);

  cudaError_t copy_to_device(unsigned char* dev_in, unsigned char* host_image, 
                                float* host_kernel,size_t image_size, int kernel_size);

  cudaError_t launch_convolution_kernels(unsigned char* dev_in, unsigned char* dev_temp, unsigned char* dev_out, 
                                            int width, int height, int kernel_size, dim3 gridSize, dim3 blockSize);

  cudaError_t copy_to_host(unsigned char* host_image, unsigned char* dev_out, size_t image_size);

  __host__ void gaussian_filter_cuda_constant(unsigned char* image, int width, int height, float sigma, int kernel_size);
  #ifdef __cplusplus
  }
  #endif


  #endif // GAUSSIAN_FILTER_CUDA_CONSTANT_H