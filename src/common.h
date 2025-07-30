#ifndef COMMON_H
#define COMMON_H

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

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

// Common data structures
typedef struct {
    int filter_choice;
    int kernel_size;
    float sigma;
    double cpu_time;
    double wall_time;
} BenchmarkResult;

#endif // COMMON_H