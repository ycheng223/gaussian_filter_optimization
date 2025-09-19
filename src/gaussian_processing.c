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