#ifndef GAUSSIAN_FILTER_CUDA_H
#define GAUSSIAN_FILTER_CUDA_H

#include "common.h"

#ifndef __NVCC__
#define __global__
#define __host__
#define __device__
#endif

// Cuda Filter implementations
__global__ void gaussian_filter_cuda_convolve(unsigned char* dev_in, unsigned char* dev_out,
    int width, int height, const float* kernel, int kernel_size, int direction);

#ifdef __cplusplus
extern "C" {
#endif

__host__ void gaussian_filter_cuda(unsigned char* image, int width, int height, float sigma, int kernel_size);

void warmup_gpu(void);



#ifdef __cplusplus
}
#endif

#endif // GAUSSIAN_FILTER_CUDA_H