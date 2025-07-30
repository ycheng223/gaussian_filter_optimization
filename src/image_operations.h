#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include "common.h"
#include "lodepng.h"

// Image operations
int image_encode(const char* filepath, const unsigned char* image_data, 
                int width, int height, int filter_choice);

void image_decode(const char* filename, int kernel_size, float sigma, 
                 int filter_choice, BenchmarkResult *result);

#endif // IMAGE_OPS_H