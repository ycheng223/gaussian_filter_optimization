#include "gaussian_filter.h"

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
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_kernel_base(image, temp, x, y, width, height, sigma, range);
        }
    }

    memcpy(image, temp, width * height * CHANNELS_PER_PIXEL);
    free(temp);
}




// Next to cut down on computational cost, we will transform the 2D Gaussian filter into the product of
// 2 1D Gaussian filters. This is possible because gaussian processes are separable and hence, any matrix
// can be represented as the product of two 1D filters.
void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    int range = kernel_size / 2;

    // First we allocate a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Horizontal pass (is_vertical = 0)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_separable_kernel(image, temp, x, y, width, height, sigma, range, 0);
        }
    }
    // Vertical pass (is_vertical = 1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_separable_kernel(temp, image, x, y, width, height, sigma, range, 1);
        }
    }
    free(temp); // release memory allocated for temp buffer
}



// Base SSE case of applying naive SSE optimization directly on the separable case above.
void gaussian_filter_sse_base(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    
    int range = kernel_size / 2;
    
    // Precompute gaussian kernel
    float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
    if(!kernel) {
        return;
    }

    // Pad the image to make it a multiple of 4 and ensure that the kernel does not exceed the boundaries
    PaddedImage* padded = image_padding_transform(image, width, height, range);
    if (!padded) {
        free(kernel);
        return;
    }

    // Get the dimensions and memory location of the padded image from the returned struct
    unsigned char* padded_image = padded->data;
    int padded_width = padded->padded_width;
    int padded_height = padded->padded_height;

    // Next we prepare a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(padded_width * padded_height * CHANNELS_PER_PIXEL);
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        free(padded->data);
        free(padded);
        free(kernel);
        return;
    }
    
    // Horizontal pass (is_vertical = 0 in process_sse_base)
    for(int y = 0; y < padded_height; y++) {
        for(int x = 0; x < padded_width; x += 4) {
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();

            process_sse_base(padded_image, kernel, x, y, padded_width, padded_height, range,
                           &sum_red, &sum_green, &sum_blue, 0);

            store_rgb_results(temp + (y * padded_width + x) * CHANNELS_PER_PIXEL,
                            sum_red, sum_green, sum_blue);
        }
    }

    // The temp buffer will also serve as our intermediate buffer since we need to process the
    // Vertical Pass on the results of our Horizontal Pass

    // Vertical pass (is_vertical = 1 in process_sse_base)
    for(int y = 0; y < padded_height; y++) {
        for(int x = 0; x < padded_width; x += 4) {
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();

            process_sse_base(temp, kernel, x, y, padded_width, padded_height, range,
                           &sum_red, &sum_green, &sum_blue, 1);

            store_rgb_results(temp + (y * padded_width + x) * CHANNELS_PER_PIXEL,
                            sum_red, sum_green, sum_blue);
        }
    }

    // After vertical pass, offset for padding and copy only the valid region back to the original image
    for(int y = range; y < height + range; y++) {
        memcpy(image + (y - range) * width * CHANNELS_PER_PIXEL,
               temp + (y * padded_width + range) * CHANNELS_PER_PIXEL,
               width * CHANNELS_PER_PIXEL);
    }

    // Clean up allocated memory
    free(temp);
    free(kernel);
    free(padded->data);
    free(padded);
}



// SSE implementation using that deinterleaves RGB in the SSE register itself using shuffle and bitmasks.
void gaussian_filter_sse_shuffle(unsigned char* image, int width, int height, float sigma, int kernel_size){

    int range = kernel_size / 2;
    
    // Precompute gaussian kernel
    float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
    if(!kernel) {
        return;
    }

    // Pad the image to make it a multiple of 4 and ensure that the kernel does not exceed the boundaries
    PaddedImage* padded = image_padding_transform(image, width, height, range);
    if (!padded) {
        free(kernel);
        return;
    }

    // Use padded dimensions directly from struct
    unsigned char* padded_image = padded->data;
    int padded_width = padded->padded_width;
    int padded_height = padded->padded_height;

    // Masks for shuffling the RGB values to deinterleave
    const __m128i mask_red = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 12, 9, 6, 3, 0, 0x80);
    const __m128i mask_green = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 13, 10, 7, 4, 1, 0x80);
    const __m128i mask_blue = _mm_set_epi8(0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 14, 11, 8, 5, 2, 0x80);

    // Next we prepare a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(padded_width * padded_height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        free(padded->data);
        free(padded);
        free(kernel);
        return;
    }
    
    // Now onto the SSE horizontal pass...
    for(int y = 0; y < padded_height; y++) {
        for(int x = 0; x < padded_width; x += 4) { // Process 4 pixels in parallel using SSE
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();

            process_sse_shuffle(padded_image, kernel, mask_red, mask_green, mask_blue,
                            x, range, padded_width, &sum_red, &sum_green, &sum_blue);

            unsigned char* temp_loc = temp + (y * padded_width + x) * CHANNELS_PER_PIXEL;
            store_rgb_results(temp_loc, sum_red, sum_green, sum_blue);
        }
    }

    // Now for the SSE Vertical Pass, because the image data is in row-major format, we will have to jump
    // between pixels to get to the next column meaning non-contiguous memory access. Therefore we will
    // TRANSPOSE a copy of the data (so that it becomes COLUMN-MAJOR) and store it seperately in memory.
    unsigned char* transposed_img = transpose_rgb_block_sse(temp, padded_width, padded_height);
    if (!transposed_img) {
        fprintf(stderr, "Failed to transpose image\n");
        free(temp);
        free(kernel);
        free(padded->data);
        free(padded);
        return;
    }

    // Create final buffer for storing results
    unsigned char* final = (unsigned char*)malloc(padded_width * padded_height * CHANNELS_PER_PIXEL);
    if (!final) {
        fprintf(stderr, "Failed to allocate final buffer\n");
        free(transposed_img);
        free(temp);
        free(kernel);
        free(padded->data);
        free(padded);
        return;
    }

    // Vertical Pass using SSE, since it is transposed, RGB is already seperated so we can process it using SSE directly
    for (int y = 0; y < padded_width; y++) {
        for (int x = 0; x < padded_height; x += 4) {
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();

            process_sse_shuffle_vertical(transposed_img, kernel, x, y, range, 
                                      padded_height, padded_width, 
                                      &sum_red, &sum_green, &sum_blue);
            
            // Store in final buffer instead of original image
            store_rgb_results(final + (y * padded_width + x) * CHANNELS_PER_PIXEL,
                            sum_red, sum_green, sum_blue);
        }
    }

    // After vertical pass, copy only the valid region back to the original image
    for(int y = range; y < height + range; y++) {
        memcpy(image + (y - range) * width * CHANNELS_PER_PIXEL,
               final + (y * padded_width + range) * CHANNELS_PER_PIXEL,
               width * CHANNELS_PER_PIXEL);
    }

    // clean up memory
    free(final);
    free(transposed_img);
    free(temp);
    free(kernel);
    free(padded->data);
    free(padded);
}