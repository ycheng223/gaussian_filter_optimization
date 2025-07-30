#include "gaussian_filter.h"

// Decode the image into 1D array of rgb values and load into a contiguous memory space
void image_decode(const char* filename, int kernel_size, float sigma, 
                int filter_choice, BenchmarkResult *result) {

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
        image_encode(output_path, image, width, height, filter_choice); // re-encode as image and save to disc in "output" folder
    }

    free(image);

    image = NULL;
}


// Encode array of RGB values back into image to get back the tranformed image after the gaussian filter is applied
// Saves seperate output images based on filter_choice so we can compare the results later
int image_encode(const char* filepath, const unsigned char* image_data, 
                int width, int height, int filter_choice) {

    // First we create an output filename using filter choice
    const char* filter_names[] = {"base", "separable", "sse_base", "sse_shuffle"};
    
    // Next we retrieve the original filename using the filepath
    const char *last_slash = strrchr(filepath, '/'); // find the last slash in the filepath
    const char *filename = last_slash ? last_slash + 1 : filepath; // if last slash exists get everything after that point (i.e. this will be the filename + .extension)
    
    // Calculate required buffer size with proper length checks
    size_t basename_len = strlen(filename);
    const char *dot = strrchr(filename, '.');
    if (dot) {
        basename_len = dot - filename;
    }
    
    // Check if we have enough space for basename + "_" + filter_name + ".png" + null
    size_t filter_name_len = strlen(filter_names[filter_choice - 1]);
    size_t required_size = basename_len + 1 + filter_name_len + 4 + 1; // +1 for "_", +4 for ".png", +1 for null
    
    // Create output filename with safe buffer size
    char output_filename[2048];
    if (required_size > sizeof(output_filename)) {
        fprintf(stderr, "Output filename would be too long (needs %zu bytes)\n", required_size);
        return -1;
    }
    
    // Copy basename and create new filename with filter type
    strncpy(output_filename, filename, basename_len);
    output_filename[basename_len] = '\0';
    snprintf(output_filename + basename_len, sizeof(output_filename) - basename_len,
             "_%s.png", filter_names[filter_choice - 1]);
    
    // Save the image (standard lodepng method)
    unsigned error = lodepng_encode24_file(output_filename, image_data, width, height);
    if(error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }
    
    printf("Saved processed image as: %s\n", output_filename);
    return 0;
}