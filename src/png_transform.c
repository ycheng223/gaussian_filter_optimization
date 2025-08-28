#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../inc/gaussian_filter.h"
#include "../inc/png_transform.h"

// Decode the image into a heap-allocated 1D array of RGBA bytes.
// Returns pointer to image buffer on success (caller must free), NULL on error.
// out_width/out_height are set when non-NULL.
unsigned char* image_decode(const char* filename, int *out_width, int *out_height) {

    unsigned error;
    unsigned char* image = NULL;
    unsigned width = 0; // lodepng will write dimensions here
    unsigned height = 0;
    char input_path[512];
    
    // Construct absolute path to input folder
    snprintf(input_path, sizeof(input_path), "./input/%s", filename);

    error = lodepng_decode32_file(&image, &width, &height, input_path);
    if(error) {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
        if(image) { free(image); image = NULL; }
        return NULL;
    }

    if (out_width)  *out_width  = (int)width;
    if (out_height) *out_height = (int)height;

    return image;
}

// Encode array of RGBA values back into image to get back the transformed image after the gaussian filter is applied
// Returns a pointer to the encoded image data (free after image is saved)
unsigned char* image_encode(const unsigned char* image_data, int width, int height, size_t* out_size) {
    unsigned char* png_data = NULL;
    unsigned error = lodepng_encode32(&png_data, out_size, image_data, width, height);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return NULL;
    }
    return png_data;
}