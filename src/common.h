#ifndef COMMON_H
#define COMMON_H

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// Debugging Libraries
#include <assert.h>

// SSE Libraries
#include <immintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

// Constants
#define CHANNELS_PER_PIXEL 3
#define MIN_SIGMA 0.5
#define MAX_SIGMA 2
#define SIGMA_STEP 0.5
#define SSE_BLOCK_SIZE 4

// Image Dimension Constants
#define PADDED_IMG_SIZE(width, height) ((width) * (height) * CHANNELS_PER_PIXEL)
#define PADDED_ROW_SIZE(width) ((width) * CHANNELS_PER_PIXEL)
#define PIXEL_OFFSET(x, y, width) (((y) * (width) + (x)) * CHANNELS_PER_PIXEL)

// Memory Index Constants
#define ROW_MAJOR_OFFSET(x, y, width) (((y) * (width) + (x)) * CHANNELS_PER_PIXEL)
#define COL_MAJOR_OFFSET(x, y, height) (((x) * (height) + (y)) * CHANNELS_PER_PIXEL)

// Common data structures
typedef struct {
    int filter_choice;
    int kernel_size;
    float sigma;
    double cpu_time;
    double wall_time;
} BenchmarkResult;

typedef struct {
    unsigned char* data;
    int padded_width;
    int padded_height;
} PaddedImage;

#endif // COMMON_H