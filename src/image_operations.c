#include "gaussian_filter.h"

// Decode the image into 1D array of rgb values and load into a contiguous memory space
void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice, BenchmarkResult *result) {

    unsigned error;
    unsigned char* image = 0;
    int width, height;
    char input_path[256];
    char output_path[256];
    
    // Construct absolute paths to input and output folders
    snprintf(input_path, sizeof(input_path), "./input/%s", filename);
    snprintf(output_path, sizeof(output_path), "./output/%s", filename);

    error = lodepng_decode24_file(&image, (unsigned*)&width, (unsigned*)&height, input_path); // decode and store image contiguously into memory as 1D array, 3 channels per pixel (i.e. RGB, NO ALPHA)
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //note: to include alpha, we would the the lodepng_decode32_file function

    else{
        measure_filter_time(image, width, height, sigma, kernel_size, filter_choice, result); // measure time and apply selected gaussian filter
        image_encode(output_path, image, width, height); // re-encode as image and save to disc in "output" folder
    }

    free(image);

    image = NULL;
}


// Encode array of RGB values back into image to get back the tranformed image after the gaussian filter is applied
void image_encode(const char* filepath, const unsigned char* image, int width, int height) {
         
    unsigned error = lodepng_encode24_file(filepath, image, (unsigned)width, (unsigned)height); //encode and save to disc as "filename"
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}
