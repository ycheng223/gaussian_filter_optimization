#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <stdint.h>
#include <emmintrin.h>
#include <tmmintrin.h>

#include "../inc/image_operations.h"
#include "../inc/gaussian_filter.h"
#include "../inc/utility.h"



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

// Helper function to load 4x4 block of RGBA into 4 SSE registers
static inline void load_block(const unsigned char* in_ptr, int width,
                                       __m128i* out_r0, __m128i* out_r1, __m128i* out_r2, __m128i* out_r3) {
    *out_r0 = _mm_loadu_si128((const __m128i*)&in_ptr[0]);
    *out_r1 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 4]);
    *out_r2 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 8]);
    *out_r3 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 12]);
}


static inline void transpose_block(
    __m128i row0, __m128i row1, __m128i row2, __m128i row3,
    __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3) {

    // This is a new way to perform the transpose directly on the SSE registers.
    // Given 4 128-bit registers, each of which holds 4 32-bit pixels (RGBA) ordered from high bit to low bit...
    //      row0: | Pixel 4  | Pixel 3  | Pixel 2  | Pixel 1  | -> | [R4 G4 B4 A4] | [R3 G3 B3 A3] | [R2 G2 B2 A2] | [R1 G1 B1 A1] |
    //      row1: | Pixel 8  | Pixel 7  | Pixel 6  | Pixel 5  | -> | [R8 G8 B8 A8] | [R7 G7 B7 A7] | [R6 G6 B6 A6] | [R5 G5 B5 A5] |
    //      row2: | Pixel 12 | Pixel 11 | Pixel 10 | Pixel 9  | -> | [R12 G12 B12 A12] | [R11 G11 B11 A11] | [R10 G10 B10 A10] | [R9 G9 B9 A9] |
    //      row3: | Pixel 16 | Pixel 15 | Pixel 14 | Pixel 13 | -> | [R16 G16 B16 A16] | [R15 G15 B15 A15] | [R14 G14 B14 A14] | [R13 G13 B13 A13] |

    // Interleave 32-bit pixels from low and high halves (64 bits) of the registers

    // Interleave 32-bit elements from the low 64 bits of the first two rows.
    __m128i tmp0 = _mm_unpacklo_epi32(row0, row1);  // | Pixel 6  | Pixel 2  | Pixel 5  | Pixel 1  | -> | [R6 G6 B6 A6] | [R2 G2 B2 A2] | [R5 G5 B5 A5] | [R1 G1 B1 A1] |
    // Do the same for the last two rows.
    __m128i tmp1 = _mm_unpacklo_epi32(row2, row3);  // | Pixel 14 | Pixel 10 | Pixel 13 | Pixel 9  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R13 G13 B13 A13] | [R9 G9 B9 A9] |

    // Interleave 32-bit elements from the high 64 bits of the first two rows.
    __m128i tmp2 = _mm_unpackhi_epi32(row0, row1);  // | Pixel 8  | Pixel 4  | Pixel 7  | Pixel 3  | -> | [R8 G8 B8 A8] | [R4 G4 B4 A4] | [R7 G7 B7 A7] | [R3 G3 B3 B3] |
    // Do the same for the last two rows.
    __m128i tmp3 = _mm_unpackhi_epi32(row2, row3);  // | Pixel 16 | Pixel 12 | Pixel 15 | Pixel 11 | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R15 G15 B15 A15] | [R11 G11 B11 A11] |

    // Summary:128i* out0, __m128i* out1, __m128i* out2, __m128i* out3) {

    // register_tmp0: | Pixel 6  | Pixel 2  | Pixel 5  | Pixel 1  | -> | [R6  G6  B6  A6]  | [R2  G2  B2  A2]  | [R5  G5  B5  A5]  | [R1  G1  B1  A1]  |
    // register_tmp1: | Pixel 14 | Pixel 10 | Pixel 13 | Pixel 9  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R13 G13 B13 A13] | [R9  G9  B9  A9]  |
    // register_tmp2: | Pixel 8  | Pixel 4  | Pixel 7  | Pixel 3  | -> | [R8  G8  B8  A8]  | [R4  G4  B4  A4]  | [R7  G7  B7  A7]  | [R3  G3  B3  B3]  |
    // register_tmp3: | Pixel 16 | Pixel 12 | Pixel 15 | Pixel 11 | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R15 G15 B15 A15] | [R11 G11 B11 A11] |


    // Interleave the low 64-bit chunks of the temporary registers together.
    *out0 = _mm_unpacklo_epi64(tmp0, tmp1); // | Pixel 13 | Pixel 9  | Pixel 5  | Pixel 1  | -> | [R13 G13 B13 A13] | [R9  G9  B9  A9]  | [R5 G5 B5 A5] | [R1 G1 B1 A1] |
    // Interleave the high 64-bit chunks.
    *out1 = _mm_unpackhi_epi64(tmp0, tmp1); // | Pixel 14 | Pixel 10 | Pixel 6  | Pixel 2  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R6 G6 B6 A6] | [R2 G2 B2 A2] |

    // Repeat for the remaining temporary registers.
    *out2 = _mm_unpacklo_epi64(tmp2, tmp3); // | Pixel 15 | Pixel 11 | Pixel 7  | Pixel 3  | -> | [R15 G15 B15 A15] | [R11 G11 B11 A11] | [R7 G7 B7 A7] | [R3 G3 B3 A3] |
    *out3 = _mm_unpackhi_epi64(tmp2, tmp3); // | Pixel 16 | Pixel 12 | Pixel 8  | Pixel 4  | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R8 G8 B8 A8] | [R4 G4 B4 A4] |
    
    // Final result is transposed 4x4 block of Pixels/RGBA (wow!):
    // register_out0: | Pixel 13 | Pixel 9  | Pixel 5  | Pixel 1  | -> | [R13 G13 B13 A13] | [R9  G9  B9  A9]  | [R5 G5 B5 A5] | [R1 G1 B1 A1] |
    // register_out1: | Pixel 14 | Pixel 10 | Pixel 6  | Pixel 2  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R6 G6 B6 A6] | [R2 G2 B2 A2] |
    // register_out2: | Pixel 15 | Pixel 11 | Pixel 7  | Pixel 3  | -> | [R15 G15 B15 A15] | [R11 G11 B11 A11] | [R7 G7 B7 A7] | [R3 G3 B3 A3] |
    // register_out3: | Pixel 16 | Pixel 12 | Pixel 8  | Pixel 4  | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R8 G8 B8 A8] | [R4 G4 B4 A4] |

}

static inline void deinterleave_block(unsigned char* out_ptr, int width, int height, int x, int y,
                                                 __m128i r_plane, __m128i g_plane, __m128i b_plane, __m128i a_plane) {
    size_t plane_size = (size_t)width * (size_t)height;
    size_t offset = (size_t)y * width + x;

    _mm_storeu_si128((__m128i*)(out_ptr + 0 * plane_size + offset), r_plane); // Store Red plane: register_r_plane: | [R1 R2 R3 R4] | [R5 R6 R7 R8] | [R9 R10 R11 R12] | [R13 R14 R15 R16]
    _mm_storeu_si128((__m128i*)(out_ptr + 1 * plane_size + offset), g_plane); // Store Green plane
    _mm_storeu_si128((__m128i*)(out_ptr + 2 * plane_size + offset), b_plane); // Store Blue plane
    _mm_storeu_si128((__m128i*)(out_ptr + 3 * plane_size + offset), a_plane); // Store Alpha plane
}

// SSE version of image transpose, transposes image in 4x4 blocks and stores seperate blocks back in memory
unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height) {

    // Allocate memory for transposed data
    unsigned char* transposed = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL);
    if (!transposed) {
        fprintf(stderr, "Failed to allocate transpose buffer\n");
        return NULL;
    }

    // Process the image in 4x4 pixel blocks
    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            __m128i r0, r1, r2, r3; // Registers for input rows
            __m128i o0, o1, o2, o3; // Registers for de-interleaved output channels
            
            unsigned char* in_ptr = input + (y * width + x) * CHANNELS_PER_PIXEL;
            
            // Load 4x4 block of RGBA (16 pixels i.e. 16*32 bits = 512 bits) into 4 SSE registers (4*128 = 512 bits)
            load_block(in_ptr, width, &r0, &r1, &r2, &r3);

            // Transpose 4x4 block using register interleave operations
            transpose_block(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
            
            // Deinterleave RGBA and store each channel in seperate registers (total of 4)
            deinterleave_block(transposed, width, height, x, y, o0, o1, o2, o3);
        }
    }

    return transposed;
}


// Helper function to convert float values from SSE processing back into 32 bit integer
static inline void convert_float_int32(__m128 r_float, __m128 g_float, __m128 b_float,
                                               __m128i* r_int, __m128i* g_int, __m128i* b_int) {
    *r_int = _mm_cvtps_epi32(r_float);
    *g_int = _mm_cvtps_epi32(g_float);
    *b_int = _mm_cvtps_epi32(b_float);
}

// Helper function to pack 32 bit integer down to 8 bit with stuation
static inline __m128i pack_int8(__m128i r_int, __m128i g_int, __m128i b_int) {
    //First pack 32 bit down to 16 bit with saturation
    __m128i rg_packed16 = _mm_packs_epi32(r_int, g_int);
    __m128i b_packed16 = _mm_packs_epi32(b_int, b_int); // Use blue for both halves as a placeholder

    // Pack 16-bit signed integers into 8-bit unsigned integers with saturation
    return _mm_packus_epi16(rg_packed16, b_packed16);
}

// Re-interleave RGB values using 4 channels/pixel offset in output image memory block
static inline void reinterleave_block(unsigned char* output, __m128i packed_rgb, const unsigned char* input) {
    unsigned char* rgb_ptr = (unsigned char*)&packed_rgb;

    for (int i = 0; i < 4; i++) {
        output[i * 4 + 0] = rgb_ptr[i];         // R channel
        output[i * 4 + 1] = rgb_ptr[i + 4];     // G channel
        output[i * 4 + 2] = rgb_ptr[i + 8];     // B channel
        output[i * 4 + 3] = input[i * 4 + 3];   // A channel (copied from original input)
    }
}


void store_rgba_results(unsigned char* output, __m128 red, __m128 green, __m128 blue, const unsigned char* input) {
    __m128i r_int, g_int, b_int;

    // Convert from float to int32
    convert_float_int32(red, green, blue, &r_int, &g_int, &b_int);

    // Pack down to int8
    __m128i packed_rgb = pack_int8(r_int, g_int, b_int);

    // Re-interleave back to RGB
    reinterleave_block(output, packed_rgb, input);
}