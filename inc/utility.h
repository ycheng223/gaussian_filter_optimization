#ifndef UTILITY_H
#define UTILITY_H

#include "common.h"
#include "gaussian_filter_cuda.h"

// Statistics and timing
void print_statistics(BenchmarkResult* results, int count);

void countdown(int seconds);

void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result);


int save_image(const char* filename, const unsigned char* image_data, 
               int width, int height, int filter_choice, int kernel_size);

#endif // UTILITY_H