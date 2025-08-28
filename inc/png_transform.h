#ifndef PNG_TRANSFORM_H
#define PNG_TRANSFORM_H

#include "common.h"
#include "lodepng/lodepng.h"

// Decode PNG image into 1D array of RGD values in memory block accessed using *result
unsigned char* image_decode(const char* filename, int *out_width, int *out_height);

// Encode back into PNG
unsigned char* image_encode(const unsigned char* image_data, int width, int height, size_t* out_size);


#endif // PNG_TRANSFORM_H
