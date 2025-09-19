#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <stdint.h>

#include "../inc/image_operations.h"
#include "../inc/gaussian_filter.h"
#include "../inc/utility.h"



//------------------------------------------------------------------------------------------------------------------
// Border Operations
//------------------------------------------------------------------------------------------------------------------

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
        int src_row_start_position = y * width * CHANNELS_PER_PIXEL; // Get the position of the start of the row in the original image (in memory i.e. 1D row-major array)
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


//------------------------------------------------------------------------------------------------------------------
// Normal image transpose
//------------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------------
// Normal image deinterleave
//------------------------------------------------------------------------------------------------------------------

// Deinterleaves an RGB image into separate R, G, and B planes.
unsigned char* deinterleave_rgb_base(unsigned char* input, int width, int height) {
    // Calculate the total number of pixels and the size of a single color plane.
    size_t num_pixels = (size_t)width * height;
    size_t plane_size = num_pixels;

    // Allocate memory for the output buffer, which will store the deinterleaved color planes.
    unsigned char* deinterleaved = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height));
    if (!deinterleaved) {
        fprintf(stderr, "Failed to allocate deinterleave buffer\n");
        return NULL;
    }

    // Deinterleave the image by separating the R, G, B, and A channels.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t index = (size_t)y * width + x;
            size_t offset = index * CHANNELS_PER_PIXEL;
            deinterleaved[index] = input[offset];                     // Red channel
            deinterleaved[index + plane_size] = input[offset + 1];     // Green channel
            deinterleaved[index + 2 * plane_size] = input[offset + 2]; // Blue channel
            deinterleaved[index + 3 * plane_size] = input[offset + 3]; // Alpha channel
        }
    }

    return deinterleaved;
}

//------------------------------------------------------------------------------------------------------------------
// Normal image reinterleave
//------------------------------------------------------------------------------------------------------------------

// Reinterleaves separate R, G, B, and A planes back into a single RGBA image. Store in memory block.

unsigned char* reinterleave_rgb_base(unsigned char* input, int width, int height) {
    // Calculate the total number of pixels and the size of a single color plane.
    size_t num_pixels = (size_t)width * height;
    size_t plane_size = num_pixels;

    // Allocate memory for the interleaved output buffer.
    unsigned char* interleaved = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height));
    if (!interleaved) {
        fprintf(stderr, "Failed to allocate reinterleave buffer\n");
        return NULL;
    }

    // Pointers to the start of each color plane in the input buffer.
    const unsigned char* r_plane = input;
    const unsigned char* g_plane = input + plane_size;
    const unsigned char* b_plane = input + 2 * plane_size;
    const unsigned char* a_plane = input + 3 * plane_size;

    // Reinterleave the color planes into the output buffer.
    for (size_t i = 0; i < num_pixels; ++i) {
        size_t offset = i * CHANNELS_PER_PIXEL;
        interleaved[offset + 0] = r_plane[i]; // Red
        interleaved[offset + 1] = g_plane[i]; // Green
        interleaved[offset + 2] = b_plane[i]; // Blue
        interleaved[offset + 3] = a_plane[i]; // Alpha
    }

    return interleaved;
}