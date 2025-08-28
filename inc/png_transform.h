#ifndef PNG_TRANSFORM_H
#define PNG_TRANSFORM_H

#include "../inc/lodepng/lodepng.h"
#include "../inc/gaussian_filter.h"

// Decode PNG image into 1D array of RGD values in memory block accessed using *result
void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice, BenchmarkResult *result);

// Encode back into PNG
int image_encode(const char* filename, const unsigned char* image_data, int width, int height, int filter_choice, int kernel_size);

#endif // PNG_TRANSFORM_H
