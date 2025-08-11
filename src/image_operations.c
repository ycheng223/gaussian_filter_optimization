#include "gaussian_filter.h"

// Decode the image into 1D array of rgb values and load into a contiguous memory space
void image_decode(const char* filename, int kernel_size, float sigma, 
                int filter_choice, BenchmarkResult *result) {

    unsigned error;
    unsigned char* image = 0;
    int width = 0; // initialize to 0, lodepng_decode24_file will write the dimensions in
    int height = 0; // same as above
    char input_path[512];
    char output_path[512];
    
    // Construct absolute paths to input and output folders
    snprintf(input_path, sizeof(input_path), "./input/%s", filename);
    snprintf(output_path, sizeof(output_path), "./output/%s", filename);

    error = lodepng_decode32_file(&image, (unsigned*)&width, (unsigned*)&height, input_path); // decode and store image contiguously into memory as 1D array, 3 channels per pixel (i.e. RGB, NO ALPHA)
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //note: to include alpha, we would the the lodepng_decode32_file function

    else{
        measure_filter_time(image, width, height, sigma, kernel_size, filter_choice, result); // measure time and apply selected gaussian filter
        image_encode(filename, image, width, height, filter_choice, kernel_size); // re-encode as image and save to disc in "output" folder
    }

    free(image);

    image = NULL;
}


// Encode array of RGB values back into image to get back the transformed image after the gaussian filter is applied
// Saves separate output images based on filter_choice so we can compare the results later
int image_encode(const char* filename, const unsigned char* image_data, 
                int width, int height, int filter_choice, int kernel_size) {

    // Output filter names
    const char* filter_names[] = {"base", "separable", "sse_base", "sse_shuffle"};

    // Ensure output directory exists
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        mkdir("output", 0700);
    }

    // Truncate the extension (i.e. ".png")
    const char *dot = strrchr(filename, '.');
    size_t basename_len = dot ? (size_t)(dot - filename) : strlen(filename);

    // Compose output filename: output/<filename>_k<kernel_size>_<filter>.png
    char output_filename[2048];
    snprintf(output_filename, sizeof(output_filename),
            "output/%.*s_k%d_%s.png",
            (int)basename_len, filename, kernel_size, filter_names[filter_choice - 1]);

    // Save the image (standard lodepng method)
    printf("Attempting to save to: %s\n", output_filename);
    unsigned error = lodepng_encode32_file(output_filename, image_data, width, height);
    if(error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    printf("Saved processed image as: %s\n", output_filename);
    return 0;
}