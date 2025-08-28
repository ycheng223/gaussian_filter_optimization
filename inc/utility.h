#ifndef UTILITY_H
#define UTILITY_H

#include "common.h"

// Statistics and timing
void print_statistics(BenchmarkResult* results, int count);
void countdown(int seconds);
int save_image(const char* filename, const unsigned char* image_data, 
               int width, int height, int filter_choice, int kernel_size);

#endif // UTILITY_H