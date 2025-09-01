#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../inc/gaussian_processing.h"
#include "../inc/image_operations.h"


// Precompute kernel weights
float* precompute_gaussian_kernel(int kernel_size, float sigma) {
    int range = kernel_size / 2;
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    
    if(!kernel) {
        fprintf(stderr, "Failed to allocate kernel buffer\n");
        return NULL;
    }


    float weight_sum = 0.0f;
    // Calculate weights and accumulate sum
    for(int i = 0; i < kernel_size; i++) {
        int kernel_offset = i - range;
        kernel[i] = expf(-(kernel_offset * kernel_offset) / (2 * sigma * sigma));
        weight_sum += kernel[i];
    }

    // Normalize the weights
    for(int i = 0; i < kernel_size; i++) {
        kernel[i] /= weight_sum;
    }

    return kernel;
}


// Base case, 2D convolution using a moving window for the kernel
void process_kernel_base(unsigned char* image, unsigned char* temp, 
                              int x, int y, int width, int height,
                              float sigma, int range) {
    // Process each color channel
    for (int c = 0; c < CHANNELS_PER_PIXEL; c++) {
        if (c == 3) {
            // Copy the alpha channel directly
            temp[ROW_MAJOR_OFFSET(x, y, width) + 3] = image[ROW_MAJOR_OFFSET(x, y, width) + 3];
            continue;
        }
        float sum = 0.0f;
        float weight_sum = 0.0f;
        for (int kernel_y = -range; kernel_y <= range; kernel_y++) {
            for (int kernel_x = -range; kernel_x <= range; kernel_x++) {
                int x_neighbor = x + kernel_x;
                int y_neighbor = y + kernel_y;
                int clamped_index = border_clamp(width, height, x_neighbor, y_neighbor);
                int clamped_x = clamped_index % width;
                int clamped_y = clamped_index / width;
                float pixel_value = image[ROW_MAJOR_OFFSET(clamped_x, clamped_y, width) + c];
                float weight = expf(-(kernel_x * kernel_x + kernel_y * kernel_y) / (2 * sigma * sigma));
                sum += pixel_value * weight;
                weight_sum += weight;
            }
        }
        temp[ROW_MAJOR_OFFSET(x, y, width) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f);
    }
}

// Seperable case
void process_separable_kernel (unsigned char* input, unsigned char* output,
                                   int x, int y, int width, int height,
                                   float sigma, int range, int is_vertical) {

    for (int c = 0; c < CHANNELS_PER_PIXEL; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        if (c == 3) {
            // Copy the alpha channel directly
            output[ROW_MAJOR_OFFSET(x, y, width) + 3] = input[ROW_MAJOR_OFFSET(x, y, width) + 3];
            continue;
        }
        
        for (int kernel_index = -range; kernel_index <= range; kernel_index++) {
            // Calculate neighbor position based on pass direction
            int neighbor_x, neighbor_y;
            if (is_vertical) { 
                neighbor_x = x; 
                neighbor_y = y + kernel_index; 
            } 
            else { 
                neighbor_x = x + kernel_index; 
                neighbor_y = y; 
            }
            int clamped_index = border_clamp(width, height, neighbor_x, neighbor_y);
            int clamped_x = clamped_index % width;
            int clamped_y = clamped_index / width;
            float pixel_value = input[ROW_MAJOR_OFFSET(clamped_x, clamped_y, width) + c];
            float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma));
            sum += pixel_value * weight;
            weight_sum += weight;
        }
        
        output[ROW_MAJOR_OFFSET(x, y, width) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f);
    }
}

void process_sse_base(unsigned char* dest, const unsigned char* src, 
                               int width, int height, const float* kernel, int range, int is_vertical) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            // Ensure we don't process past the image width. This is safe because the image is padded.
            if (x + 15 >= width) continue;

            // Accumulators for 16 pixels (processed as 4 groups of 4 floats)
            __m128 sum_ps[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
            
            for (int k = -range; k <= range; k++) {
                const unsigned char* pixel_ptr;
                if (is_vertical) {
                    // Vertical pass: neighbor pixels are in different rows
                    pixel_ptr = src + (y + k) * width + x;
                } else {
                    // Horizontal pass: neighbor pixels are in the same row
                    pixel_ptr = src + y * width + x + k;
                }
                
                __m128i pixels_u8 = _mm_loadu_si128((const __m128i*)pixel_ptr);
                __m128 weight_vec = _mm_set1_ps(kernel[k + range]);

                // Unpack 16 uchars to 4 vectors of 4 floats
                __m128i zero = _mm_setzero_si128();
                __m128i p16_lo = _mm_unpacklo_epi8(pixels_u8, zero);
                __m128i p16_hi = _mm_unpackhi_epi8(pixels_u8, zero);
                
                __m128 p32_0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(p16_lo, zero));
                __m128 p32_1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(p16_lo, zero));
                __m128 p32_2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(p16_hi, zero));
                __m128 p32_3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(p16_hi, zero));

                // Multiply by kernel weight and accumulate
                sum_ps[0] = _mm_add_ps(sum_ps[0], _mm_mul_ps(p32_0, weight_vec));
                sum_ps[1] = _mm_add_ps(sum_ps[1], _mm_mul_ps(p32_1, weight_vec));
                sum_ps[2] = _mm_add_ps(sum_ps[2], _mm_mul_ps(p32_2, weight_vec));
                sum_ps[3] = _mm_add_ps(sum_ps[3], _mm_mul_ps(p32_3, weight_vec));
            }
            
            // Convert accumulator floats back to 32-bit integers
            __m128i i32_0 = _mm_cvtps_epi32(sum_ps[0]);
            __m128i i32_1 = _mm_cvtps_epi32(sum_ps[1]);
            __m128i i32_2 = _mm_cvtps_epi32(sum_ps[2]);
            __m128i i32_3 = _mm_cvtps_epi32(sum_ps[3]);

            // Pack 32-bit integers down to 16-bit, then to 8-bit with saturation
            __m128i i16_01 = _mm_packs_epi32(i32_0, i32_1);
            __m128i i16_23 = _mm_packs_epi32(i32_2, i32_3);
            __m128i final_u8 = _mm_packus_epi16(i16_01, i16_23);
            
            // Store the 16 processed pixels back to the destination plane
            _mm_storeu_si128((__m128i*)(dest + y * width + x), final_u8);
        }
    }
}

// To properly prepare the loop for SSE, we will need to deinterleave the RGB values in order to process them using SIMD 
// (i.e. can't have same instruction on different RGB channels).

// Specifically we will be using the SSE3 intrinsic _mm_shuffle_epi8 which will allow us to not only map any byte on a 
// SOURCE register to any byte on a DESTINATION register but also zero any of those bytes out if needed through bit masking.

// This means we can load three registers with bit masks and apply each of them to the data stored in __m128i input_data 
// (where input_data is the SOURCE register). Each bitmask maps to an individual color (RGB) and will zero out all values in input_data 
// except the values mapped to that color (i.e. every third value starting from 0 for Red, every third value starting from 1 for Blue, etc). All three masked outputs will then
// be stored in 3 additional 128 bit registers with each of these registers only containing intensity values for a single color (these are the DESTINATION registers). 
// This ensures that we can apply the gaussian filter operations both accurately and with parallelism.

void process_sse_shuffle (unsigned char* padded_image, 
                                    float* kernel, const __m128i mask_red, 
                                    const __m128i mask_green, const __m128i mask_blue, int x, int y, int range, 
                                    int padded_width, __m128* sum_red, __m128* sum_green, __m128* sum_blue) {
    
    assert((x + 3) < padded_width); // Check to see nothing goes out of bounds
                                        
    for (int k = -range; k <= range; k++) { 
        
        int row = y;  // Current row in padded coordinates
        int col = x + range + k;  // Current column in padded coordinates
        int data_offset = ROW_MAJOR_OFFSET(col, row, padded_width);

        // load 16 bytes from the padded image into SSE register at location specified by k + offset
        __m128i input_data = _mm_loadu_si128(
            (__m128i*)&padded_image[data_offset]);

        // Initialilze 128-bit registers to store masks per the rules above (i.e. 0s everywhere but positions in 
        // input_data that contain the color specified by the mask)
        __m128i red_epi_8 = _mm_shuffle_epi8(input_data, mask_red);
        __m128i green_epi_8 = _mm_shuffle_epi8(input_data, mask_green);
        __m128i blue_epi_8 = _mm_shuffle_epi8(input_data, mask_blue);

        // Convert to 32-bit integers
        __m128i red_epi_32 = _mm_cvtepu8_epi32(red_epi_8);
        __m128i green_epi_32 = _mm_cvtepu8_epi32(green_epi_8);
        __m128i blue_epi_32 = _mm_cvtepu8_epi32(blue_epi_8);

        // Convert to floats
        __m128 red_ps = _mm_cvtepi32_ps(red_epi_32);
        __m128 green_ps = _mm_cvtepi32_ps(green_epi_32);
        __m128 blue_ps = _mm_cvtepi32_ps(blue_epi_32);

        // Apply gaussian weights
        __m128 weight = _mm_set1_ps(kernel[k + range]);
        *sum_red = _mm_add_ps(*sum_red, _mm_mul_ps(red_ps, weight));
        *sum_green = _mm_add_ps(*sum_green, _mm_mul_ps(green_ps, weight));
        *sum_blue = _mm_add_ps(*sum_blue, _mm_mul_ps(blue_ps, weight));
    }
}

void process_sse_shuffle_vertical (unsigned char* transposed, float* kernel,
                                     int x, int y, int range, int height, int width,
                                     __m128* sum_red, __m128* sum_green, __m128* sum_blue) {
    
    for (int k = -range; k <= range; k++) {
        int y_offset = y + k;
        
        // Boundary check
        if (y_offset >= 0 && y_offset < width) {
            // Since data is transposed, RGB channels are already separated
            __m128 red_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[ROW_MAJOR_OFFSET(x, y_offset, height)])
            ));
            __m128 green_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[ROW_MAJOR_OFFSET(x + height, y_offset, height)])
            ));
            __m128 blue_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[ROW_MAJOR_OFFSET(x + height * 2, y_offset, height)])
            ));

            // Apply gaussian weights
            __m128 weight = _mm_set1_ps(kernel[k + range]);
            *sum_red = _mm_add_ps(*sum_red, _mm_mul_ps(red_ps, weight));
            *sum_green = _mm_add_ps(*sum_green, _mm_mul_ps(green_ps, weight));
            *sum_blue = _mm_add_ps(*sum_blue, _mm_mul_ps(blue_ps, weight));
        }
    }
}