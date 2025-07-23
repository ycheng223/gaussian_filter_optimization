#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

// Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Additional Libraries
#include "lodepng.h" // image (png only) encoder and decoder optimized to balance speed and ease-of-use


// Constants
#define CHANNELS_PER_PIXEL 3  // RGB but NO ALPHA
#define MAX_SIGMA 2 // Maximum value sigma we want to benchmark to

// Data Structures
typedef struct {
    int filter_choice;
    int kernel_size;
    float sigma;
    double cpu_time;
    double wall_time;
} BenchmarkResult; // For storing benchmark results and parameters per iteration



// Function declarations

void image_encode(const char* filepath, const unsigned char* image, int width, int height);
// Encode 1D array of RGB values in memory back into image and save this new tranformed image into "output" folder

void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice, BenchmarkResult *result);
// Load image from "input" folder, decode, and store decoded rgb data contiguously into memory as 1D array, 3 channels per pixel (i.e. RGB, NO ALPHA)

int border_clamp(int width, int height, int x, int y);
// Helper function to deal with the edges of image -> if kernel extends beyond the edges 
// of the image, fill it with the pixel value of the nearest edge and update the image data

void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size);
// apply 2D gaussian filter on image

void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size);
// apply 2D gaussian filter by decomposing it into the product of 2 1D gaussian filters

void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result);
// measure the cpu time and wall time required to apply the gaussian filter

#endif // GAUSSIAN_FILTER_H