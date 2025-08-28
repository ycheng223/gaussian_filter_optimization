#ifndef IMAGE_OPERATIONS_H
#define IMAGE_OPERATIONS_H

#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include "gaussian_filter.h"

// Padding and border operations
PaddedImage* image_padding_transform(unsigned char* image, int width, int height, int range);
int border_clamp(int width, int height, int x, int y);

// Transpose and RGB operations
unsigned char* transpose_rgb_base(unsigned char* input, int width, int height);
unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height);
void store_rgb_results(unsigned char* output, __m128 red, __m128 green, __m128 blue, const unsigned char* input);

// Statistics and timing
void print_statistics(BenchmarkResult* results, int count);
void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result);
void countdown(int seconds);

#endif // IMAGE_OPERATIONS_H