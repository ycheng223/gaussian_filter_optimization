#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include "common.h"

// Filter implementations
void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size);

void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size);

void gaussian_filter_sse_base(unsigned char* image, int width, int height, float sigma, int kernel_size);

void gaussian_filter_sse_shuffle(unsigned char* image, int width, int height, float sigma, int kernel_size);

void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result);

#endif // GAUSSIAN_FILTER_H