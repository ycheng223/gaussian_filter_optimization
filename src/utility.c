#include "gaussian_filter.h"
#include "utility.h"



// Add padding to the image fo account for boundary overlap by the sliding kernel and ensure the image is divisble by four
// Padded values are the clamped values at the closest border
PaddedImage* image_padding_transform(unsigned char* image, int width, int height, int range) {

    // First calculate the padding needed to offset range (i.e. the sliding kernel exceeds the image boundary)
    int padded_width_initial = width + range * 2;
    int padded_height_initial = height + range * 2;

    // Is it divisible by 16?
    int width_remainder = padded_width_initial % 16;
    int height_remainder = padded_height_initial % 16;

    // If yes then no need to pad more (0), if not pad with just enough to make it divisible (i.e. 16 - remainder)
    int extra_width = width_remainder ? (16 - width_remainder) : 0;
    int extra_height = height_remainder ? (16 - height_remainder) : 0;

    // Final padded dimensions
    int padded_width = padded_width_initial + extra_width;
    int padded_height = padded_height_initial + extra_height;
    
    unsigned char* padded_img = (unsigned char*)malloc(PADDED_IMG_SIZE(padded_width, padded_height));
    if (!padded_img) {
        fprintf(stderr, "Failed to allocate padded image buffer\n");
        return NULL;
    }

    // Copy the original image into the padded image row by row directly from memory, offsetting for padding
    for(int y = 0; y < height; y++) { // For each row in the image...
        int src_row_start_position = PADDED_ROW_SIZE(y * width); // Get the position of the start of the row in the original image (in memory i.e. 1D row-major array)
        int dest_row_start_position = ROW_MAJOR_OFFSET(range, y + range, padded_width); // offset for padded rows on top of image and left
        memcpy(padded_img + dest_row_start_position, 
                image + src_row_start_position, 
                PADDED_ROW_SIZE(width)); // Copy the original row into mapped memory locations of the padded image
    }

    // Pad the left and right edges of the image
    for (int y = range; y < padded_height - range; y++) { // For each row in the padded image, skipping the top and bottom padded rows
        memset(
            padded_img + PADDED_ROW_SIZE(y * padded_width),
            padded_img[ROW_MAJOR_OFFSET(range, y, padded_width)],
            PADDED_ROW_SIZE(range)
        ); // Fill the entire left edge of that row with the first pixel of the row
        memset(
            padded_img + ROW_MAJOR_OFFSET(width + range, y, padded_width),
            padded_img[ROW_MAJOR_OFFSET(width + range - 1, y, padded_width)],
            PADDED_ROW_SIZE(range)
        ); // Fill the entire right edge of that row with the last pixel of the row
    }

    // Now pad the top and bottom edges of the image by copying the first and last rows respectively
    for (int y = 0; y < range; y++) { // start from the top of the padded image and go down to where the padding ends
        memcpy(
            padded_img + PADDED_ROW_SIZE(y * padded_width),
            padded_img + PADDED_ROW_SIZE(range * padded_width),
            PADDED_ROW_SIZE(padded_width)
        ); // Fill top border by copying the first row of the image
    }

    // Fill bottom border by copying the last valid row
    for (int y = 0; y < range; y++) { // start from the bottom of the padded image and go up...
        memcpy(
            padded_img + PADDED_ROW_SIZE((padded_height - 1 - y) * padded_width),
            padded_img + PADDED_ROW_SIZE((padded_height - 1 - range) * padded_width),
            PADDED_ROW_SIZE(padded_width)
        ); // Same logic as the top border except we copy the last row of the image
    }

    // Finally save the dimensions in a struct (so we don't ahve to calculate it again) and return the struct
    PaddedImage* padded_dimensions = (PaddedImage*)malloc(sizeof(PaddedImage));
    if (!padded_dimensions) {
        free(padded_img);
        return NULL;
    }

    padded_dimensions->data = padded_img;
    padded_dimensions->padded_width = padded_width;
    padded_dimensions->padded_height = padded_height;

    return padded_dimensions;
}

// Helper function to deal with the edges of image -> if kernel extends beyond the edges
// of the image, fill it with the pixel value of the nearest edge and update the image data
int border_clamp(int width, int height, int x, int y) {
    int clamped_x = x;
    int clamped_y = y;

    if(clamped_x < 0) clamped_x = 0;
    if(clamped_x >= width) clamped_x = width - 1;

    if(clamped_y < 0) clamped_y = 0;
    if(clamped_y >= height) clamped_y = height - 1;

    return clamped_y * width + clamped_x; // Returns index of the nearest edge pixel for a 1D row-major array
}


// transposes image and seperates it into seperate blocks of r, g, and b in memory
unsigned char* transpose_rgb_base(unsigned char* input, int width, int height) {
    unsigned char* transposed = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height));
    if (!transposed) {
        fprintf(stderr, "Failed to allocate transpose buffer\n");
        return NULL;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < CHANNELS_PER_PIXEL; ++c) {
                transposed[COL_MAJOR_OFFSET(x, y, height) + c] =
                    input[ROW_MAJOR_OFFSET(x, y, width) + c];
            }
        }
    }
    return transposed;
}

// SSE version of the above, transposes in 4x4 blocks
// currently doesn't work as sse block reads past the buffer on the edges of the image
unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height) {

    // Allocate memory for transposed data
    unsigned char* transposed = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL);
    if (!transposed) {
        fprintf(stderr, "Failed to allocate transpose buffer\n");
        return NULL;
    }

    // Transpose the data in 4x4 blocks using SSE
        for (int y = 0; y < height; y += 4) {
            for (int x = 0; x < width; x += 4) {
                __m128i row0, row1, row2, row3;
                
                // Calculate input/output positions
                unsigned char* in_ptr = input + (y * width + x) * CHANNELS_PER_PIXEL; // offset memory start position from initial input image pointer
                unsigned char* out_ptr = transposed + (x * height + y) * CHANNELS_PER_PIXEL; // transposed offset memory start position (i.e. y becomes x and vice versa)
                
                // Load 4 rows of RGB data into 4 SSE registeres at in one go
                row0 = _mm_loadu_si128((__m128i*)&in_ptr[0]); // load first 16 bytes of data from memory position specified by in_ptr
                row1 = _mm_loadu_si128((__m128i*)&in_ptr[width * 3]); // second 16 bytes...
                row2 = _mm_loadu_si128((__m128i*)&in_ptr[width * 6]); // and so on...
                row3 = _mm_loadu_si128((__m128i*)&in_ptr[width * 9]);

                // Process all 4 rows
                for (int i = 0; i < 4; i++) {
                    __m128i current_row;
                    switch(i) {
                        case 0: current_row = row0; break;
                        case 1: current_row = row1; break;
                        case 2: current_row = row2; break;
                        case 3: current_row = row3; break;
                    }

                    // Initialize 3 SSE registers to store shuffle masks for RGB separation
                    __m128i mask_red = _mm_set_epi8(0x80,0x80,0x80,0x80, 12,9,6,3,0, 0x80,0x80,0x80,0x80,0x80,0x80,0x80); // Result: [0 0 0 0 R4 R3 R2 R1 R0 0 0 0 0 0 0 0]
                    __m128i mask_green = _mm_set_epi8(0x80,0x80,0x80,0x80, 13,10,7,4,1, 0x80,0x80,0x80,0x80,0x80,0x80,0x80); // Same as above we are only adding greens (offset 1)
                    __m128i mask_blue = _mm_set_epi8(0x80,0x80,0x80,0x80, 14,11,8,5,2, 0x80,0x80,0x80,0x80,0x80,0x80,0x80); // Same as above except for blues (offset 2)

                    // Initialize 3 more SSE registers to apply the shuffle on the current row to seperate RGB and store
                    __m128i red_vals = _mm_shuffle_epi8(current_row, mask_red);
                    __m128i green_vals = _mm_shuffle_epi8(current_row, mask_green);
                    __m128i blue_vals = _mm_shuffle_epi8(current_row, mask_blue);

                    // Store transposed data, offsetting for each row, by stacking the registers on 
                    // top of each other, we create the transposed columns.
                    unsigned char* block_ptr = out_ptr + i * CHANNELS_PER_PIXEL * height;
                    _mm_storeu_si128((__m128i*)block_ptr, red_vals); // store R values
                    _mm_storeu_si128((__m128i*)(block_ptr + height * CHANNELS_PER_PIXEL), green_vals);
                    _mm_storeu_si128((__m128i*)(block_ptr + 2 * height * CHANNELS_PER_PIXEL), blue_vals);
                }
            }
        }

        return transposed;
}


// Convert float values from SSE processing back into integer, pack down 8 bits (0-255)
// Re-interleave the RGB so we can rebuild the image
void store_rgb_results(unsigned char* output, __m128 red, __m128 green, __m128 blue) {
    
    // Convert float values back to integers
    __m128i red_int = _mm_cvtps_epi32(red);
    __m128i green_int = _mm_cvtps_epi32(green);
    __m128i blue_int = _mm_cvtps_epi32(blue);

    // Pack 32-bit integers into 16-bit integers with saturation
    __m128i rg_packed = _mm_packs_epi32(red_int, green_int);
    __m128i bb_packed = _mm_packs_epi32(blue_int, blue_int);

    // Pack 16-bit integers into 8-bit unsigned integers with saturation
    __m128i rgb_packed = _mm_packus_epi16(rg_packed, bb_packed); // This is a 1D array with layout [R0 R1 .... G0 G1 .... B0 B1 ....]

    // Extract the individual bytes and interleave RGB values
    unsigned char* rgb_ptr = (unsigned char*)&rgb_packed; // points to register containing 1D array above
    for (int i = 0; i < 4; i++) {
        output[i * 3] = rgb_ptr[i];          // insert ith value (1,2,3,4 -> these are R) into the correct positions in the interleaved RGB output (i.e. every 3rd position starting from 0)
        output[i * 3 + 1] = rgb_ptr[i + 4];  // insert i+4th value (5,6,7,8 -> these are G) into ""
        output[i * 3 + 2] = rgb_ptr[i + 8];  // same as above except will be blue values that are interleaved
    }
}


// Takes in data from Benchmark results and uses to compute summary statistics for each approach
void print_statistics(BenchmarkResult* results, int count) {
    double base_cpu_total = 0.0, base_wall_total = 0.0;
    double sep_cpu_total = 0.0, sep_wall_total = 0.0;
    double sse_cpu_total = 0.0, sse_wall_total = 0.0;
    double shuffle_cpu_total = 0.0, shuffle_wall_total = 0.0;
    int base_count = 0, sep_count = 0, sse_count = 0, shuffle_count = 0;


    // Print total times for each filter_choice
    for (int i = 0; i < count; i++) {
        switch(results[i].filter_choice) {
            case 1: // Base
                base_cpu_total += results[i].cpu_time;
                base_wall_total += results[i].wall_time;
                base_count++;
                break;
            case 2: // Separable
                sep_cpu_total += results[i].cpu_time;
                sep_wall_total += results[i].wall_time;
                sep_count++;
                break;
            case 3: // SSE Base
                sse_cpu_total += results[i].cpu_time;
                sse_wall_total += results[i].wall_time;
                sse_count++;
                break;
            case 4: // SSE Shuffle
                shuffle_cpu_total += results[i].cpu_time;
                shuffle_wall_total += results[i].wall_time;
                shuffle_count++;
                break;
        }
    }

    // Print average times for each filter_choice
    printf("\nAverage Times:\n");
    if (base_count > 0)
        printf("Base:     CPU: %.3fms, Wall: %.3fms\n", 
               base_cpu_total/base_count, base_wall_total/base_count);
    if (sep_count > 0)
        printf("Separable: CPU: %.3fms, Wall: %.3fms\n", 
               sep_cpu_total/sep_count, sep_wall_total/sep_count);
    if (sse_count > 0)
        printf("SSE Base:  CPU: %.3fms, Wall: %.3fms\n", 
               sse_cpu_total/sse_count, sse_wall_total/sse_count);
    if (shuffle_count > 0)
        printf("SSE Shuffle: CPU: %.3fms, Wall: %.3fms\n", 
               shuffle_cpu_total/shuffle_count, shuffle_wall_total/shuffle_count);
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
        printf("Using Base SSE Sep. Gaussian Filter (Base SSE)...\n");
        gaussian_filter_sse_base(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 4) {
        printf("Using SSE Load Shuffle Sep. Gaussian Filter...\n");
        gaussian_filter_sse_shuffle(image, width, height, sigma, kernel_size);
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