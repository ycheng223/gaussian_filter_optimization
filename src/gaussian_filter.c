#include "../inc/gaussian_filter.h"
#include "../inc/gaussian_processing.h"
#include "../inc/image_operations.h"
#include "../inc/utility.h"
#include <string.h>


void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size){

    // We are replacing each individual pixel in the image with the weighted average of it and its neighboring pixels in a nxn matrix (gaussian kernel) where n = kernel_size
    // Kernel weights are normally calculated by applying eqn for gaussian distribution to each coordinate on the gaussian kernel relative to it's center.
    // i.e. { [e^-(x^2 + y^2)] / (2*sigma^2) } where {x,y} = [0,0], [0,1], [1,0], [1,1] .... [n/2,n/2] -> note that it is n/2 because the distance is relative to the center of the kernel.
    // Get the weighted average by by multiplying the RGB value of each pixel overlayed by the gaussian kernel (i.e. dot product) and calculating their weighted average (i.e. (sum of dot products)/(weighted_sum))
    // The greater the blur (i.e. variance/sigma) the larger the gaussian kernel needs to be to maintain precision but more on that later...

    int range = kernel_size / 2;
    // First we allocate a temp buffer for the convolution
    unsigned char* temp = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height)); //total dimension will be width * height * 4 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_kernel_base(image, temp, x, y, width, height, sigma, range);
        }
    }

    memcpy(image, temp, PADDED_IMG_SIZE(width, height));
    free(temp);
}




// Next to cut down on computational cost, we will transform the 2D Gaussian filter into the product of
// 2 1D Gaussian filters. This is possible because gaussian processes are separable and hence, any matrix
// can be represented as the product of two 1D filters.
void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    int range = kernel_size / 2;

    // First we allocate a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height)); //total dimension will be width * height * 3 channels (i.e. RGB)
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



// SSE base implementation, first deinterleaves the image into separate R, G, B, and A planes. SSE operations are then applied 
// to each color plane independently. Finally, the planes are reinterleaved to form the final image.
void gaussian_filter_sse_base(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    int range = kernel_size / 2;

    // Precompute the 1D Gaussian kernel
    float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
    if (!kernel) return;

    // Pad the original interleaved image to handle kernel borders
    PaddedImage* padded = image_padding_transform(image, width, height, range);
    if (!padded) {
        free(kernel);
        return;
    }

    // Get dimensions of padded unit and allocation block size
    int padded_width = padded->padded_width;
    int padded_height = padded->padded_height;
    size_t padded_plane_size = (size_t)padded_width * padded_height;

    // Deinterleave the padded image into separate R,G,B,A planes
    unsigned char* planar_padded = deinterleave_rgb_base(padded->data, padded_width, padded_height);
    if (!planar_padded) {
        free(kernel);
        free(padded->data);
        free(padded);
        return;
    }

    // Temp buffer for horizontal pass result
    unsigned char* temp_planar_padded = (unsigned char*)malloc(padded_plane_size * CHANNELS_PER_PIXEL);
    if (!temp_planar_padded) {
        fprintf(stderr, "Failed to allocate temp planar buffer\n");
        free(kernel);
        free(padded->data);
        free(padded);
        free(planar_padded);
        return;
    }

    // Process each color plane (R, G, B) with two separable SSE passes. The Alpha plane (channel 3) is skipped.
    for (int c = 0; c < 3; ++c) {
        unsigned char* src_plane = planar_padded + c * padded_plane_size;
        unsigned char* temp_plane = temp_planar_padded + c * padded_plane_size;
        
        // Horizontal pass (reads from src_plane, writes to temp_plane)
        process_sse_base(temp_plane, src_plane, padded_width, padded_height, kernel, range, 0);
        
        // Vertical pass (reads from temp_plane, writes back to src_plane)
        process_sse_base(src_plane, temp_plane, padded_width, padded_height, kernel, range, 1);
    }

    // 6. Reinterleave the blurred R, G, B planes and the original A plane back into an RGBA image
    unsigned char* blurred_interleaved_padded = reinterleave_rgb_base(planar_padded, padded_width, padded_height);
    if (!blurred_interleaved_padded) {
        // Handle cleanup if reinterleave fails
        free(kernel);
        free(padded->data);
        free(padded);
        free(planar_padded);
        free(temp_planar_padded);
        return;
    }

    // 7. Copy the unpadded center from the result back to the original image buffer
    for (int y = 0; y < height; ++y) {
        memcpy(image + (size_t)y * width * CHANNELS_PER_PIXEL,
               blurred_interleaved_padded + ROW_MAJOR_OFFSET(range, y + range, padded_width),
               (size_t)width * CHANNELS_PER_PIXEL);
    }

    // 8. Clean up all allocated memory
    free(kernel);
    free(padded->data);
    free(padded);
    free(planar_padded);
    free(temp_planar_padded);
    free(blurred_interleaved_padded);
}

// SSE implementation using that deinterleaves RGB in the SSE register itself using shuffle and bitmasks.
void gaussian_filter_sse_shuffle(unsigned char* image, int width, int height, float sigma, int kernel_size){

    int range = kernel_size / 2;
    
    // Precompute gaussian kernel
    float* kernel = precompute_gaussian_kernel(kernel_size, sigma);
    if(!kernel) {
        return;
    }

    // Pad the image to make it a multiple of 16 and ensure that the kernel does not exceed the boundaries
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
    const __m128i mask_red   = _mm_set_epi8(0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 0);
    const __m128i mask_green = _mm_set_epi8(0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 1);
    const __m128i mask_blue  = _mm_set_epi8(0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 2);


    // Next we prepare a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
        unsigned char* temp = (unsigned char*)malloc(PADDED_IMG_SIZE(padded_width, padded_height));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        free(padded->data);
        free(padded);
        free(kernel);
        return;
    }
    
    // Shuffled SSE Horizontal pass
    for(int y = 0; y < padded_height; y++) {
        for(int x = 0; x < padded_width; x += 4) { // Four pixels at a time... (i.e. 4 * 32 bit single-precision/cycle)
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();
            process_sse_shuffle(padded_image, kernel, mask_red, mask_green, mask_blue, x, y, range, padded_width, &sum_red, &sum_green, &sum_blue);
            store_rgba_results(temp + ROW_MAJOR_OFFSET(x, y, padded_width), sum_red, sum_green, sum_blue, padded_image + ROW_MAJOR_OFFSET(x, y, padded_width));
        }
    }

    // Shuffled SSE Vertical pass
    // Because the image data is in row-major format, we will have to jump
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

    // Also allocate memory block on heap for the final output image which will use another transpose to bring it back to row-major
    unsigned char* final_transposed = (unsigned char*)malloc(PADDED_IMG_SIZE(padded_width, padded_height));
    if (!final_transposed) {
        fprintf(stderr, "Failed to allocate final buffer\n");
        free(transposed_img);
        free(temp);
        free(kernel);
        free(padded->data);
        free(padded);
        return;
    }

    // Now apply the vertical pass to the transposed image
    for (int y = 0; y < padded_width; y += 4) {
        for (int x = 0; x < padded_height; x += 4) {
            __m128 sum_red = _mm_setzero_ps();
            __m128 sum_green = _mm_setzero_ps();
            __m128 sum_blue = _mm_setzero_ps();
            process_sse_shuffle_vertical(transposed_img, kernel, x, y, range, padded_height, padded_width, &sum_red, &sum_green, &sum_blue);

            // Save result (both passes complete) to final output image
            store_rgba_results(final_transposed + COL_MAJOR_OFFSET(x, y, padded_height), sum_red, sum_green, sum_blue, transposed_img + COL_MAJOR_OFFSET(x, y, padded_height));
        }
    }

    unsigned char* final_row_major = transpose_rgb_block_sse(final_transposed, padded_height, padded_width); // Transpose again to return to row-major for output image
    
    for (int y = 0; y < height; y++) {
        memcpy(image + (size_t)y * width * CHANNELS_PER_PIXEL, final_row_major + ROW_MAJOR_OFFSET(range, y + range, padded_width), (size_t)width * CHANNELS_PER_PIXEL);
    }

    free(final_row_major);
    free(final_transposed);
    free(transposed_img);
    free(temp);
    free(kernel);
    free(padded->data);
    free(padded);
}