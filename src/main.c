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

/// Measure wall time (absolute time) and CPU time (computational time) needed to finish applying the gaussian filter to the image
void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result) {

    clock_t start_cpu, end_cpu;
    time_t start_wall, end_wall;
    double cpu_time_used, wall_time_used;
    
    printf("\n=== Processing Image ===\n");
    printf("Image size: %dx%d pixels\n", width, height);
    printf("Kernel size: %d\n", kernel_size);
    printf("Sigma: %.2f\n", sigma);
    
    start_cpu = clock();
    start_wall = time(NULL);
    
    if (filter_choice == 1) {
        printf("Using 2D Gaussian Filter (Base)...\n");
        gaussian_filter_base(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 2) {
        printf("Using Separable Gaussian Filter...\n");
        gaussian_filter_separable(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 3) {
        printf("Using Seperable Gaussian Filter (SSE)...\n");
        gaussian_filter_sse(image, width, height, sigma, kernel_size);
    } else {
        fprintf(stderr, "Invalid filter choice: %d\n", filter_choice);
        return;
    }
    
    end_cpu = clock();
    end_wall = time(NULL);
    
    cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    wall_time_used = difftime(end_wall, start_wall);

    result->cpu_time = cpu_time_used;
    result->wall_time = wall_time_used;
    
    printf("CPU time: %.4f seconds\n", cpu_time_used);
    printf("Wall time: %.4f seconds\n", wall_time_used);
    printf("=== Processing Complete ===\n\n");
}

unsigned char* image_padding_transform(unsigned char* image, int width, int height, int range) {
    int padded_width = width + range * 2;
    int padded_height = height + range * 2;
    
    unsigned char* padded_img = (unsigned char*)malloc(padded_width * padded_height * CHANNELS_PER_PIXEL);
    if (!padded_img) {
        fprintf(stderr, "Failed to allocate padded image buffer\n");
        return NULL;
    }

    // Copy the original image into the padded image row by row directly from memory, offsetting for padding
    for(int y = 0; y < height; y++) { // For each row in the image...
        int src_row_start_position = y * width * CHANNELS_PER_PIXEL; // Get the position of the start of the row in the original image (in memory i.e. 1D row-major Array)
        int dest_row_start_position = (y + range) // offset for padded rows on top of image
                                        * padded_width // row-major offset, this also accounts for the padding on the right side of the image
                                        + (range) // offset for padded rows on the left of image
                                        * CHANNELS_PER_PIXEL; // 3 RGB values per pixel

        memcpy(padded_img + dest_row_start_position, 
                image + src_row_start_position, 
                width * CHANNELS_PER_PIXEL); // Copy the original row into mapped memory locations of the padded image
    }

    // Pad the left and right edges of the image
    for (int y = range; y < padded_height - range; y++) { // For each row in the padded image, skipping the top and bottom padded rows
        memset(
            padded_img + y * padded_width * CHANNELS_PER_PIXEL,
            padded_img[y * padded_width * CHANNELS_PER_PIXEL + range * CHANNELS_PER_PIXEL],
            range * CHANNELS_PER_PIXEL
        ); // Fill the entire left edge of that row with the first pixel of the row
        memset(
            padded_img + (y * padded_width + width + range) * CHANNELS_PER_PIXEL,
            padded_img[(y * padded_width + width + range - 1) * CHANNELS_PER_PIXEL],
            range * CHANNELS_PER_PIXEL
        ); // Fill the entire right edge of that row with the last pixel of the row
    }

    // Now pad the top and bottom edges of the image by copying the first and last rows respectively
    for (int y = 0; y < range; y++) { // start from the top of the padded image and go down to where the padding ends
        memcpy(
            padded_img + y * padded_width * CHANNELS_PER_PIXEL,
            padded_img + range * padded_width * CHANNELS_PER_PIXEL,
            padded_width * CHANNELS_PER_PIXEL
        ); // Fill top border by copying the first row of the image
    }

    // Fill bottom border by copying the last valid row
    for (int y = 0; y < range; y++) { // start from the bottom of the padded image and go up...
        memcpy(
            padded_img + (padded_height - 1 - y) * padded_width * CHANNELS_PER_PIXEL,
            padded_img + (padded_height - 1 - range) * padded_width * CHANNELS_PER_PIXEL,
            padded_width * CHANNELS_PER_PIXEL
        ); // Same logic as the top border except we copy the last row of the image
    }

    return padded_img;
}

// Helper function to deal with the edges of image -> if kernel extends beyond the edges 
// of the image, fill it with the pixel value of the nearest edge and update the image data
int border_clamp(int width, int height, int x, int y){
    int clamped_x = x;
    int clamped_y = y;

    if(clamped_x < 0) clamped_x = 0;
    if(clamped_x >= width) clamped_x = width - 1;

    if(clamped_y < 0) clamped_y = 0;
    if(clamped_y >= height) clamped_y = height - 1;

    return clamped_y * width + clamped_x; // Returns index of the nearest edge pixel for a 1D row-major array
}

void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size){

    // We are replacing each individual pixel in the image with the weighted average of it and its neighboring pixels in a nxn matrix (gaussian kernel) where n = kernel_size
    // Kernel weights are normally calculated by applying eqn for gaussian distribution to each coordinate on the gaussian kernel relative to it's center.
    // i.e. { [e^-(x^2 + y^2)] / (2*sigma^2) } where {x,y} = [0,0], [0,1], [1,0], [1,1] .... [n/2,n/2] -> note that it is n/2 because the distance is relative to the center of the kernel.
    // Get the weighted average by by multiplying the RGB value of each pixel overlayed by the gaussian kernel (i.e. dot product) and calculating their weighted average (i.e. (sum of dot products)/(weighted_sum))
    // The greater the blur (i.e. variance/sigma) the larger the gaussian kernel needs to be to maintain precision but more on that later...

    int range = kernel_size / 2;
    // First we allocate a temp buffer for the convolution
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    for (int y = 0; y < height; y++) { // For every row...
        for (int x = 0; x < width; x++) { // For every pixel...
            for (int c = 0; c < 3; c++) { // For every RGB channel... (this is basically every value in the 1D image array)

                float sum = 0.0f; // Initialize a weighted sum used to calculate weighted average
                float weight_sum = 0.0f; // Initialize a sum of weights to calculate weighted average


                // Now we iterate through every element in the n x n kernel and calculate it's weight based on gaussian distribution
                // For each weight, multiply by it's corresponding pixel value in the image data and use these values to update
                // The weighted sum and sum of weights for this gaussian kernel
                for (int kernel_y = -range; kernel_y <= range; kernel_y++) { // for every row in the kernel...
                    for (int kernel_x = -range; kernel_x <= range; kernel_x++) { //for every column in the kernel...
                        int x_neighbor = x + kernel_x; // absolute x position of neighboring pixel
                        int y_neighbor = y + kernel_y; // absolute y position of neighboring pixel
                        int clamped_index = border_clamp(width, height, x_neighbor, y_neighbor); // clamp pixel index to index of nearest edge if kernel position exceeds edge 
                        float pixel_value = image[3 * clamped_index + c]; // start of pixel's RGB values in the 1D array of image data stored in memory (hence, 3 * clamped_index + c)
                        float weight = expf(-(kernel_x * kernel_x + kernel_y * kernel_y) / (2 * sigma * sigma)); // calculate gaussian weights ie. [e^(x^2 + y^2)] / (2 * sigma^2)
                        sum += pixel_value * weight; // iteratively sum dot products of pixel_rgb and gaussian weights to get numerator of weighted avg
                        weight_sum += weight; // iteratively sum gaussian weights to get the denominator
                    }

                }
                temp[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); // weighted average = (sum of dot products) / sum of gaussian weights), clamp between 0 and 255
            }
        }
    }
    memcpy(image, temp, width * height * 3); // copy contents of temp buffer (i.e. blurred pixels) back to image buffer
    free(temp); // release memory allocated for temp buffer
}


void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size){
    int range = kernel_size / 2;
    
    // Next to cut down on computational cost, we will transform the 2D Gaussian filter into the product of
    // 2 1D Gaussian filters. This is possible because gaussian processes are separable and hence, any matrix
    // can be represented as the product of two 1D filters.

    // First we allocate a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Horizontal pass: apply 1D Gaussian kernel to each row
    for(int y = 0; y < height; y++){ // For every row...
        for(int x = 0; x < width; x++){ // For every pixel....
            for(int c = 0; c < 3; c++){ //For every RGB channel....
                float sum = 0.0f; // weighted sum to calculate blur for pixel in center of kernel window
                float weight_sum = 0.0f; // Sum of weights for normalization -> calculate weighted average
                for (int kernel_index = -range; kernel_index <= range; kernel_index++){ //For each relative pixel position (to center of kernel) in the kernel window...
                    int x_neighbor = x + kernel_index; // get the absolute index of the pixel
                    int clamped_index = border_clamp(width, height, x_neighbor, y); //clamp x_neighbor if kernel extends beyond edge of image
                    float pixel_value = image[3 * clamped_index + c]; //Get neighbor pixel value for current channel
                    float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma)); //Calculate 1D Gaussian weight
                    sum += pixel_value * weight; //Multiply and accumulate
                    weight_sum += weight; //Accumulate weights for normalization
                }
                temp[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); //Normalize and store result
            }
        }
    }
    
    // Vertical pass: apply 1D Gaussian kernel to each column of the horizontally filtered result
    for(int y = 0; y < height; y++){ // For every row...
        for(int x = 0; x < width; x++){ // And every pixel....
            for(int c = 0; c < 3; c++){ //And every RGB channel....
                float sum = 0.0f; // weighted sum to calculate blur for pixel in center of kernel window
                float weight_sum = 0.0f; // Sum of weights for normalization -> calculate weighted average
                for (int kernel_index = -range; kernel_index <= range; kernel_index++){ //For each pixel index in the kernel window...
                    int y_neighbor = y + kernel_index; // get the absolute index of the pixel
                    int clamped_index = border_clamp(width, height, x, y_neighbor); //clamp y_neighbor if kernel extends beyond edge of image
                    float pixel_value = temp[3 * clamped_index + c]; //Get neighbor pixel value from temp buffer (horizontally filtered)
                    float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma)); //Calculate 1D Gaussian weight (same as horizontal)
                    sum += pixel_value * weight; //Multiply and accumulate
                    weight_sum += weight; //Accumulate weights for normalization
                }
                image[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); //Normalize and store result back to image
            }
        }
    }
    
    free(temp); // release memory allocated for temp buffer
}

void gaussian_filter_sse(unsigned char* image, int width, int height, float sigma, int kernel_size){

    // We will need to deinterleave the RGB values in order to process them paralelly using SIMD (i.e. can't have same instruction on different RGB channels).
    // Specifically we will be using the SSE3 intrinsic _mm_shuffle_epi8 which will allow us to not only map any byte on a SOURCE 
    // register to any byte on a DESTINATION register but also zero any of those bytes out if needed through bit masking.

    // This means we can load three registers with bit masks and apply each of them
    // to the data stored in __m128i input_data (where input_data is the SOURCE register). 
    // Each bitmask maps to an individual color (RGB) and will zero out all values in input_data 
    // except the values mapped to that color (i.e. every third value starting from 0 for Red,
    // every third value starting from 1 for Blue, etc). All three masked outputs will then
    // be stored in 3 additional 128 bit registers with each of these registers only containing
    // intensity values for a single color (these are the DESTINATION registers). 
    // This ensures that we can apply the gaussian filter operations both accurately and with parallelism.

    // Initialilze 128-bit registers to store masks per the rules above (i.e. 0s everywhere but positions in input_data that 
    // contain the color specified by the mask)

    const __m128i mask_red = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 12, 9, 6, 3, 0, 0x80);
    const __m128i mask_green = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 13, 10, 7, 4, 1, 0x80);
    const __m128i mask_blue = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 14, 11, 8, 5, 2, 0x80);
    
    int range = kernel_size / 2;
    
    // To further cut down on computational cost, precompute the 1D gaussian kernel weights.

    // First we will pre-calculate the 1D Gaussian kernel and store it in a buffer
    float* kernel = (float*)malloc(kernel_size * sizeof(float)); // allocate memory for the 1D Gaussian kernel
    if(!kernel) {
        fprintf(stderr, "Failed to allocate kernel buffer\n");
        return;
    }
    float weight_sum = 0.0f; // Sum of weights for normalization -> calculate weighted average
    for(int i = 0; i < kernel_size; i++){
        int kernel_offset = i - range; // calculate the offset from the center of the kernel
        kernel[i] = expf(-(kernel_offset * kernel_offset) / (2 * sigma * sigma)); // Calculate 1D Gaussian weight
        weight_sum += kernel[i]; // Accumulate weights for normalization
    }
    // Normalize the kernel weights
    for(int i = 0; i < kernel_size; i++){
        kernel[i] /= weight_sum; 
    }

    unsigned char* padded_image = image_padding_transform(image, width, height, range); // Allocate memory for the padded image and calculate its dimensions
    if (!padded_image) {
        free(kernel);
        return;
    }
    int padded_width = width + 2 * range;

    // Next we prepare a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Now onto the SSE horizontal pass...
    for(int y = 0; y < height; y++) { // For every row...
        for(int x = 0; x < width; x += 5){ // We are processing 16 bytes at a time to align with SSE register size (i.e. 5 pixels + 1 bit = 3*5 + 1 = 16)
            __m128 sum_red = _mm_setzero_ps(); // Initialize 3 128 bit SSE registers on the CPU (i.e. XMM0, XMM1 ... etc), we will hold 4 single-precision (32 bit) FP values on each register
            __m128 sum_green = _mm_setzero_ps(); // One register for each color (RGB)
            __m128 sum_blue = _mm_setzero_ps();

            for(int k = 0; k < range; k++){ // For every element in the kernel...
                int data_offset = 3*(x + range + k) * 3*(range + k) * padded_width; // adjust for data offset on the padded image
                
                __m128i input_data = _mm_loadu_si128(
                    (__m128i*)&padded_image[data_offset]); // load 16 bytes from the padded image into SSE register at location specified by k + offset

                // Initialize more registers and apply the bitmask using _mm_shuffle_epi8.
                __m128i red_epi_8 = _mm_shuffle_epi8(input_data, mask_red);
                __m128i green_epi_8 = _mm_shuffle_epi8(input_data, mask_green);
                __m128i blue_epi_8 = _mm_shuffle_epi8(input_data, mask_blue);

                // Because output is typed as 8-bit integers after the shuffle operation (_mm_shuffle_epi8 operates exclusively at byte granularity), 
                // we need to convert it back to 32 bit floating point.

                // For speed, using cvtepu8 to first convert to 32 bit integers (note that epi_32 stands for 32-bit extended packed integer).
                __m128i red_epi_32 = _mm_cvtepu8_epi32(red_epi_8);
                __m128i green_epi_32 = _mm_cvtepu8_epi32(green_epi_8);
                __m128i blue_epi_32 = _mm_cvtepu8_epi32(blue_epi_8);

                // Then convert to 32-bit floats (note that "_ps" stands for packed single precision)
                __m128 red_ps = _mm_cvtepi32_ps(red_epi_32);
                __m128 green_ps = _mm_cvtepi32_ps(green_epi_32);
                __m128 blue_ps = _mm_cvtepi32_ps(blue_epi_32);

                // Now we apply the gaussian filter operations; broadcast weight to all 4 pixel values stored in a single register, then multiply and accumulate.
                __m128 weight = _mm_set1_ps(kernel[k + range]); // SSE intrinsic to broadcast a single fp value to all elements
                // For each color, multiply the pixel values by the weight and add to the running sum.
                sum_red = _mm_add_ps(sum_red, _mm_mul_ps(red_ps, weight));
                sum_green = _mm_add_ps(sum_green, _mm_mul_ps(green_ps, weight));
                sum_blue = _mm_add_ps(sum_blue, _mm_mul_ps(blue_ps, weight));
            }

            // Store to temp buffer so we can access during vertical pass
            store_rgb_results(temp + (x * width + y) * CHANNELS_PER_PIXEL,
                sum_red, sum_green, sum_blue);
        }
    }
    
    // Now for the SSE Vertical Pass, because the image data is in row-major format, we will have to jump
    // between pixels to get to the next column meaning non-contiguous memory access. Therefore we will
    // TRANSPOSE a copy of the data (so that it becomes COLUMN-MAJOR) and store it seperately in memory.

    // First allocate memory for the transposed data...
    unsigned char* transposed = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL);
    if (!transposed) {
        fprintf(stderr, "Failed to allocate transpose buffer\n");
        free(temp);
        free(kernel);
        return;
    }

    // Transpose the data in 4x4 blocks so that it fits perfectly into 3 SSE registers 
    // i.e. 4 * 4 * 3 = 48 bytes = 3 * 16 bytes (SSE registers are 128 bits = 16 bytes)
    // It also fits into the CPU cache line (64 bytes > 48 bytes)
    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            transpose_rgb_block_sse(
                temp + (y * width + x) * CHANNELS_PER_PIXEL,
                transposed + (x * height + y) * CHANNELS_PER_PIXEL,
                width, height, 4
            );
        }
    }

    // Vertical Pass using SSE, since it is transposed, RGB is already seperated so we can process it using SSE directly
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x += 4) {
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();

            for (int k = -range; k <= range; k++) {
                int y_offset = y + k;
                if (y_offset >= 0 && y_offset < width) {

                    // Load the data directly into registers!
                    __m128 red_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x) * CHANNELS_PER_PIXEL])
                    ));
                    __m128 green_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x + 1) * CHANNELS_PER_PIXEL])
                    ));
                    __m128 blue_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x + 2) * CHANNELS_PER_PIXEL])
                    ));

                    // Multiply and accumulate
                    __m128 weight = _mm_set1_ps(kernel[k + range]);
                    sum_red = _mm_add_ps(sum_red, _mm_mul_ps(red_ps, weight));
                    sum_green = _mm_add_ps(sum_green, _mm_mul_ps(green_ps, weight));
                    sum_blue = _mm_add_ps(sum_blue, _mm_mul_ps(blue_ps, weight));
                }
            }

            // Store results
            store_rgb_results(image + (x * width + y) * CHANNELS_PER_PIXEL,
                            sum_red, sum_green, sum_blue);
        }
    }
    free(transposed);
    free(temp);
    free(kernel);
    free(padded_image);
}

void transpose_rgb_block_sse(unsigned char* input, unsigned char* output, int width, int height, int block_size) {

    __m128i row0, row1, row2, row3;
    
    // Load 4 rows of RGB data (12 bytes each)
    row0 = _mm_loadu_si128((__m128i*)&input[0]);
    row1 = _mm_loadu_si128((__m128i*)&input[width * 3]);
    row2 = _mm_loadu_si128((__m128i*)&input[width * 6]);
    row3 = _mm_loadu_si128((__m128i*)&input[width * 9]);

    // Shuffle masks for RGB separation
    __m128i mask_red = _mm_set_epi8(0x80,0x80,0x80,0x80, 12,9,6,3, 0x80,0x80,0x80,0x80, 8,5,2,0);
    __m128i mask_green = _mm_set_epi8(0x80,0x80,0x80,0x80, 13,10,7,4, 0x80,0x80,0x80,0x80, 9,6,3,1);
    __m128i mask_blue = _mm_set_epi8(0x80,0x80,0x80,0x80, 14,11,8,5, 0x80,0x80,0x80,0x80, 10,7,4,2);

    // Separate channels and store
    __m128i red_vals = _mm_shuffle_epi8(row0, mask_red);
    __m128i green_vals = _mm_shuffle_epi8(row0, mask_green);
    __m128i blue_vals = _mm_shuffle_epi8(row0, mask_blue);

    // Store transposed data
    _mm_storeu_si128((__m128i*)&output[0], red_vals);
    _mm_storeu_si128((__m128i*)&output[width], green_vals);
    _mm_storeu_si128((__m128i*)&output[width * 2], blue_vals);
}

void store_rgb_results(unsigned char* output, __m128 red, __m128 green, __m128 blue){
    //placeholder to re-interleave the rgb values
}

void countdown(int seconds){
    time_t start_time = time(NULL); // get start time
    time_t current_time;
    int remaining_time = seconds;

    while(remaining_time > 0){
        printf("\r%d\n ", remaining_time);
        do{
            current_time = time(NULL); // repeatedly update current_time and wait...
        } while(current_time == start_time); // until current_time != start_time and...
        
        start_time = current_time; // update start time and wait once more until current time again no longer equals start
        remaining_time--; // repeat until remaining time goes down to 0
    }
    printf("\n----------Starting Test----------\n");
}


/*---------------------------------------------MAIN----------------------------------------------------*/

int main(){

    // Iteratively benchmark each filtering technique and record the parameters (technique, sigma, kernel_size, cpu_time, wall_time) in an array
    int techniques[] = {1, 2, 3}; // 1 = base, 2 = seperable, 3 = SSE
    int n_techniques = sizeof(techniques) / sizeof(techniques[0]);
    int total_results = n_techniques * 8; // total # of iterations =  n_techniques*(max_sigma - starting_sigma)/step_size + 1 = n_techniques*(4 - 0.5)/0.5 + 1 = n_techniques*8
    BenchmarkResult *results = malloc(total_results * sizeof(BenchmarkResult)); // Total # iterations * size of struct = total bytes of memory we need to allocate
    if(!results){
        fprintf(stderr, "Memory Allocation Failed!\n");
        return 1;
    }

    printf("\nThis program will compare the computational efficiency of different\n");
    printf("Gaussian filtering techniques by applying successively larger Gaussian\n");
    printf("kernels onto a test image and measuring the computational time\n");
    printf("\n\n\nPress ENTER to begin...\n");
    getchar();

    printf("\nStarting test in.....\n");
    countdown(3);

    int n_sigma_steps = (int)((MAX_SIGMA - MIN_SIGMA) / SIGMA_STEP); // given min, max, and step size of sigma, get total number of times we run the benchmark
    int filter_choice;
    int result_idx = 0;
    for(int choice = 0; choice < n_techniques; choice++){ // For each filtering technique...
        filter_choice = techniques[choice];
        for(int i = 1; i <= n_sigma_steps; i++){
            float sigma = MIN_SIGMA + i * SIGMA_STEP; // Calculate the current sigma value
            int kernel_size = 2*(int)ceil(3*sigma) + 1; // First, calculate an acceptable kernel size given the sigma level
            BenchmarkResult result;  // Instantiate struct BenchmarkResult for holding processing times
            
            // Fill with initial parameters
            result.filter_choice = filter_choice;
            result.sigma = sigma;
            result.kernel_size = kernel_size;

            image_decode("test_1.png", kernel_size, sigma, filter_choice, &result); // Decode and initialize benchmark, update result
            results[result_idx++] = result; //fill array with updated results
            }
        }
    printf("All Tests Complete!\n");
    printf("-----\t-----\t-----\t--------\t---------\n");
    

    // Print the results into a table:
    printf("\n=== Benchmark Results ===\n");
    printf("%-10s %-8s %-8s %-12s %-12s\n", "Filter", "Kernel", "Sigma", "CPU Time", "Wall Time");
    printf("%-10s %-8s %-8s %-12s %-12s\n", "------", "------", "-----", "--------", "----------");

    for (int i = 0; i < result_idx; i++) {
        const char* filter_name = (results[i].filter_choice == 1) ? "Base" : "Separable";
        printf("%-10s %-8d %-8.1f %-12.4f %-12.4f\n",
            filter_name,
            results[i].kernel_size,
            results[i].sigma,
            results[i].cpu_time,
            results[i].wall_time
        );
    }

    // Calculate total and average computational times for each Gaussian filtering technique
    double base_cpu_total = 0, base_wall_total = 0;
    double sep_cpu_total = 0, sep_wall_total = 0;
    int base_count = 0, sep_count = 0;

    for (int i = 0; i < result_idx; i++) {
        if (results[i].filter_choice == 1) {
            base_cpu_total += results[i].cpu_time;
            base_wall_total += results[i].wall_time;
            base_count++;
        } else {
            sep_cpu_total += results[i].cpu_time;
            sep_wall_total += results[i].wall_time;
            sep_count++;
        }
    }

    printf("\n=== Total Times ===\n");
    printf("Base Filter - Total CPU: %.4f, Total Wall: %.4f\n", 
        base_cpu_total, base_wall_total);
    printf("Separable Filter - Total CPU: %.4f, TOtal Wall: %.4f\n", 
        sep_cpu_total, sep_wall_total);


    printf("\n=== Average Times ===\n");
    printf("Base Filter - Average CPU: %.4f, Average Wall: %.4f\n", 
        base_cpu_total / base_count, base_wall_total / base_count);
    printf("Separable Filter - Average CPU: %.4f, Average Wall: %.4f\n", 
        sep_cpu_total / sep_count, sep_wall_total / sep_count);
    printf("\nTotal tests completed: %d\n", result_idx);
    free(results);
}