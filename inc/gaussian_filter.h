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
#include <unistd.h> // standard library for switching directories


// Constants
#define CHANNELS_PER_PIXEL 3  // RGB but NO ALPHA


// Function declarations

void image_encode(const char* filename, const unsigned char* image, unsigned width, unsigned height);
// Encode 1D array of RGB values in memory back into image and save this new tranformed image into "output" folder

void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice);
// Load image from "input" folder, decode, and store decoded rgb data contiguously into memory as 1D array, 3 channels per pixel (i.e. RGB, NO ALPHA)

int border_clamp(const unsigned char* image, unsigned width, unsigned height, int x, int y);
// Helper function to deal with the edges of image -> if kernel extends beyond the edges 
// of the image, fill it with the pixel value of the nearest edge and update the image data

void gaussian_filter_base(unsigned char* image, unsigned width, unsigned height, float sigma, int kernel_size);
//apply 2D gaussian filter on image

void gaussian_filter_separable(unsigned char* image, unsigned width, unsigned height, float sigma, int kernel_size);
//apply 2D gaussian filter by decomposing it into the product of 2 1D gaussian filters

void compare_filter_performance(unsigned char* image, unsigned width, unsigned height, float sigma, int kernel_size);
// Compare processing time between 2D and separable Gaussian filters


#endif // GAUSSIAN_FILTER_H