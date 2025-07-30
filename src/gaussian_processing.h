#ifndef GAUSSIAN_PROCESSING_H
#define GAUSSIAN_PROCESSING_H

#include "common.h"

// Processing functions

float* precompute_gaussian_kernel(int kernel_size, float sigma);

void process_kernel_base(unsigned char* image, unsigned char* temp, int x, int y, int width, int height, float sigma, int range);

void process_separable_kernel(unsigned char* input, unsigned char* output, int x, int y, int width, int height, float sigma, int range, int is_vertical);

void process_sse_base(unsigned char* input, float* kernel, int x, int y, int width, int height, int range, __m128* sum_red, __m128* sum_green, __m128* sum_blue, int is_vertical);

void process_sse_shuffle(unsigned char* padded_image, float* kernel, const __m128i mask_red, const __m128i mask_green, const __m128i mask_blue, int x, int range, int padded_width, __m128* sum_red, __m128* sum_green, __m128* sum_blue);

void process_sse_shuffle_vertical(unsigned char* transposed, float* kernel, int x, int y, int range, int height, int width, __m128* sum_red, __m128* sum_green, __m128* sum_blue);

#endif // GAUSSIAN_PROCESSING_H