#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include "common.h"
#include "lodepng.h"

// Image operations
void image_encode(const char* filepath, const unsigned char* image, int width, int height);
void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice, BenchmarkResult *result);

#endif // IMAGE_OPS_H