#ifndef UTILITY_H
#define UTILITY_H

#include "common.h"

// Utility functions
unsigned char* image_padding_transform(unsigned char* image, int width, int height, int range);

unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height);

int border_clamp(int width, int height, int x, int y);

void store_rgb_results(unsigned char* output, __m128 red, __m128 green, __m128 blue);

void print_statistics(BenchmarkResult* results, int count);

void countdown(int seconds);

#endif // UTILITY_H