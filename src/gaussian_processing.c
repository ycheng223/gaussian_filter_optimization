#include "gaussian_filter.h"


// Precompute kernel weights
static float* precompute_gaussian_kernel(int kernel_size, float sigma) {
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
static void process_kernel_base(unsigned char* image, unsigned char* temp, 
                              int x, int y, int width, int height,
                              float sigma, int range) {
    // Process each color channel
    for (int c = 0; c < CHANNELS_PER_PIXEL; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Process each position in the kernel window
        for (int kernel_y = -range; kernel_y <= range; kernel_y++) {
            for (int kernel_x = -range; kernel_x <= range; kernel_x++) {
                int x_neighbor = x + kernel_x;
                int y_neighbor = y + kernel_y;
                int clamped_index = border_clamp(width, height, x_neighbor, y_neighbor);
                float pixel_value = image[3 * clamped_index + c];
                float weight = expf(-(kernel_x * kernel_x + kernel_y * kernel_y) / (2 * sigma * sigma));
                sum += pixel_value * weight;
                weight_sum += weight;
            }
        }
        
        temp[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f);
    }
}


// Seperable case
static void process_separable_kernel (unsigned char* input, unsigned char* output,
                                   int x, int y, int width, int height,
                                   float sigma, int range, int is_vertical) {

    for (int c = 0; c < CHANNELS_PER_PIXEL; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
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
            float pixel_value = input[3 * clamped_index + c];
            float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma));
            
            sum += pixel_value * weight;
            weight_sum += weight;
        }
        
        output[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f);
    }
}


static void process_sse_base(unsigned char* input, float* kernel,
                           int x, int y, int width, int height, int range,
                           __m128* sum_red, __m128* sum_green, __m128* sum_blue,
                           int is_vertical) {
    
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
        
        // Load RGB data
        __m128i input_data = _mm_loadu_si128((__m128i*)&input[clamped_index * CHANNELS_PER_PIXEL]);
        
        // Convert to floats
        __m128 red_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(input_data));
        __m128 green_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(input_data, 1)));
        __m128 blue_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(input_data, 2)));

        // Apply gaussian weights
        __m128 weight = _mm_set1_ps(kernel[kernel_index + range]);
        *sum_red = _mm_add_ps(*sum_red, _mm_mul_ps(red_ps, weight));
        *sum_green = _mm_add_ps(*sum_green, _mm_mul_ps(green_ps, weight));
        *sum_blue = _mm_add_ps(*sum_blue, _mm_mul_ps(blue_ps, weight));
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

static void process_sse_shuffle (unsigned char* padded_image, 
                                    float* kernel, const __m128i mask_red, 
                                    const __m128i mask_green, const __m128i mask_blue, int x, int range, 
                                    int padded_width, __m128* sum_red, __m128* sum_green, __m128* sum_blue) {
    
    for(int k = 0; k < range; k++) { 
        int data_offset = 3*(x + range + k) * 3*(range + k) * padded_width;

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

static void process_sse_shuffle_vertical (unsigned char* transposed, float* kernel,
                                     int x, int y, int range, int height, int width,
                                     __m128* sum_red, __m128* sum_green, __m128* sum_blue) {
    
    for (int k = -range; k <= range; k++) {
        int y_offset = y + k;
        
        // Boundary check
        if (y_offset >= 0 && y_offset < width) {
            // Since data is transposed, RGB channels are already separated
            __m128 red_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x) * CHANNELS_PER_PIXEL])
            ));
            __m128 green_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x + height) * CHANNELS_PER_PIXEL])
            ));
            __m128 blue_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
                _mm_loadu_si128((__m128i*)&transposed[(y_offset * height + x + height * 2) * CHANNELS_PER_PIXEL])
            ));

            // Apply gaussian weights
            __m128 weight = _mm_set1_ps(kernel[k + range]);
            *sum_red = _mm_add_ps(*sum_red, _mm_mul_ps(red_ps, weight));
            *sum_green = _mm_add_ps(*sum_green, _mm_mul_ps(green_ps, weight));
            *sum_blue = _mm_add_ps(*sum_blue, _mm_mul_ps(blue_ps, weight));
        }
    }
}