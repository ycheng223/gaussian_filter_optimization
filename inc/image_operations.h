#ifndef IMAGE_OPERATIONS_H
#define IMAGE_OPERATIONS_H

#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include "common.h"

// Padding and border clamping operations
PaddedImage* image_padding_transform(unsigned char* image, int width, int height, int range);
int border_clamp(int width, int height, int x, int y);

// Transpose, Deinterleave, and RGB operations
unsigned char* transpose_rgb_base(unsigned char* input, int width, int height);
unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height);

// Store to output memory block
void store_rgba_results(unsigned char* output, __m128 red, __m128 green, __m128 blue, const unsigned char* input);


#endif // IMAGE_OPERATIONS_H
