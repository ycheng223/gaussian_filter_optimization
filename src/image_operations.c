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

//------------------------------------------------------------------------------------------------------------------
// SSE image transpose
//------------------------------------------------------------------------------------------------------------------

// Helper function to load 4x4 block of RGBA into 4 SSE registers
static inline void load_block(const unsigned char* in_ptr, int width,
                                       __m128i* out_r0, __m128i* out_r1, __m128i* out_r2, __m128i* out_r3) {
    *out_r0 = _mm_loadu_si128((const __m128i*)&in_ptr[0]);
    *out_r1 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 4]);
    *out_r2 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 8]);
    *out_r3 = _mm_loadu_si128((const __m128i*)&in_ptr[width * 12]);
}

// Helper function to transpose block directly on SSE registers through repeated low-high interleaving of register pairs containing the column values
static inline void transpose_block(
    __m128i row0, __m128i row1, __m128i row2, __m128i row3, // rows of 4x4 block saved into 4 SSE registers (4 pixels/register)
    __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3) {

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

    // Register blocks after these steps:
    // register_tmp0: | Pixel 6  | Pixel 2  | Pixel 5  | Pixel 1  | -> | [R6  G6  B6  A6]  | [R2  G2  B2  A2]  | [R5  G5  B5  A5]  | [R1  G1  B1  A1]  |
    // register_tmp1: | Pixel 14 | Pixel 10 | Pixel 13 | Pixel 9  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R13 G13 B13 A13] | [R9  G9  B9  A9]  |
    // register_tmp2: | Pixel 8  | Pixel 4  | Pixel 7  | Pixel 3  | -> | [R8  G8  B8  A8]  | [R4  G4  B4  A4]  | [R7  G7  B7  A7]  | [R3  G3  B3  B3]  |
    // register_tmp3: | Pixel 16 | Pixel 12 | Pixel 15 | Pixel 11 | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R15 G15 B15 A15] | [R11 G11 B11 A11] |


    // Interleave the low 64-bit chunks of the first 2 temporary registers together (1 and 2) and save to output register.
    *out0 = _mm_unpacklo_epi64(tmp0, tmp1);
    // | Pixel 13 | Pixel 9  | Pixel 5  | Pixel 1  | -> | [R13 G13 B13 A13] | [R9  G9  B9  A9]  | [R5 G5 B5 A5] | [R1 G1 B1 A1] |
    // Interleave the high 64-bit chunks.
    *out1 = _mm_unpackhi_epi64(tmp0, tmp1);
    // | Pixel 14 | Pixel 10 | Pixel 6  | Pixel 2  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R6 G6 B6 A6] | [R2 G2 B2 A2] |

    // Repeat for the remaining temporary registetrs (3 and 4).
    *out2 = _mm_unpacklo_epi64(tmp2, tmp3);
    // | Pixel 15 | Pixel 11 | Pixel 7  | Pixel 3  | -> | [R15 G15 B15 A15] | [R11 G11 B11 A11] | [R7 G7 B7 A7] | [R3 G3 B3 A3] |
    *out3 = _mm_unpackhi_epi64(tmp2, tmp3);
    // | Pixel 16 | Pixel 12 | Pixel 8  | Pixel 4  | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R8 G8 B8 A8] | [R4 G4 B4 A4] |
    
    // Final result is transposed 4x4 block of Pixels/RGBA (wow!):
    // register_out0: | Pixel 13 | Pixel 9  | Pixel 5  | Pixel 1  | -> | [R13 G13 B13 A13] | [R9  G9  B9  A9]  | [R5 G5 B5 A5] | [R1 G1 B1 A1] |
    // register_out1: | Pixel 14 | Pixel 10 | Pixel 6  | Pixel 2  | -> | [R14 G14 B14 A14] | [R10 G10 B10 A10] | [R6 G6 B6 A6] | [R2 G2 B2 A2] |
    // register_out2: | Pixel 15 | Pixel 11 | Pixel 7  | Pixel 3  | -> | [R15 G15 B15 A15] | [R11 G11 B11 A11] | [R7 G7 B7 A7] | [R3 G3 B3 A3] |
    // register_out3: | Pixel 16 | Pixel 12 | Pixel 8  | Pixel 4  | -> | [R16 G16 B16 A16] | [R12 G12 B12 A12] | [R8 G8 B8 A8] | [R4 G4 B4 A4] |

}

static inline void store_transposed_block(unsigned char* out_ptr, __m128i col0, __m128i col1, __m128i col2, __m128i col3) {
    _mm_storeu_si128((__m128i*)(out_ptr), col0);
    _mm_storeu_si128((__m128i*)(out_ptr + 16), col1);
    _mm_storeu_si128((__m128i*)(out_ptr + 32), col2);
    _mm_storeu_si128((__m128i*)(out_ptr + 48), col3);
}

// SSE version of image transpose, transposes image in 4x4 blocks and stores seperate blocks back in memory
unsigned char* transpose_rgb_block_sse(unsigned char* input, int width, int height) {

    // Allocate memory for transposed data
    unsigned char* transposed_block = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL);
    if (!transposed_block) {
        fprintf(stderr, "Failed to allocate transpose buffer\n");
        return NULL;
    }

    // Process the image in 4x4 pixel blocks
    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            __m128i r0, r1, r2, r3; // Registers for input rows
            __m128i col0, col1, col2, col3; // Registers to store transposed columns
            
            unsigned char* in_ptr = input + (y * width + x) * CHANNELS_PER_PIXEL;
            unsigned char* out_ptr = transposed_block + ((size_t)y * width + x) * CHANNELS_PER_PIXEL;
            
            // Load 4x4 block of RGBA (16 pixels i.e. 16*32 bits = 512 bits) into 4 SSE registers (4*128 = 512 bits)
            load_block(in_ptr, width, &r0, &r1, &r2, &r3);

            // Transpose 4x4 block using register interleave operations
            transpose_block(r0, r1, r2, r3, &col0, &col1, &col2, &col3);

            // Store into the transposed memory block
            store_transposed_block(out_ptr, col0, col1, col2, col3);
        }
    }
    return transposed_block;
}


//------------------------------------------------------------------------------------------------------------------
// SSE Retranspose function to convert transposed image block (4x4) held in SSE registers back to original using vector operations (i.e. non-symmetric)
//------------------------------------------------------------------------------------------------------------------

static inline void load_transposed_block(const unsigned char* in_ptr, __m128i* out_col0, __m128i* out_col1, __m128i* out_col2, __m128i* out_col3) {
    *out_col0 = _mm_loadu_si128((const __m128i*)(in_ptr));
    *out_col1 = _mm_loadu_si128((const __m128i*)(in_ptr + 16));
    *out_col2 = _mm_loadu_si128((const __m128i*)(in_ptr + 32));
    *out_col3 = _mm_loadu_si128((const __m128i*)(in_ptr + 48));
}

static inline void store_interleaved_block(unsigned char* out_ptr, int width, __m128i r0, __m128i r1, __m128i r2, __m128i r3) {
    _mm_storeu_si128((__m128i*)(out_ptr), r0);
    _mm_storeu_si128((__m128i*)(out_ptr + (size_t)width * 4), r1);
    _mm_storeu_si128((__m128i*)(out_ptr + (size_t)width * 8), r2);
    _mm_storeu_si128((__m128i*)(out_ptr + (size_t)width * 12), r3);
}


// Inverse of the above to recreate the image (for debug testing of transpose)
unsigned char* retranspose_rgb_block_sse(unsigned char* input, int width, int height) {
    unsigned char* retransposed = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if (!retransposed) {
        fprintf(stderr, "Failed to allocate retranspose buffer\n");
        return NULL;
    }

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            __m128i r0, r1, r2, r3;
            __m128i col0, col1, col2, col3;
            
            unsigned char* in_ptr = input + ((size_t)y * width + x) * CHANNELS_PER_PIXEL;
            unsigned char* out_ptr = retransposed + ((size_t)y * width + x) * CHANNELS_PER_PIXEL;

            load_transposed_block(in_ptr, &col0, &col1, &col2, &col3);
            transpose_block(col0, col1, col2, col3, &r0, &r1, &r2, &r3);
            store_interleaved_block(out_ptr, width, r0, r1, r2, r3);
        }
    }
    return retransposed;
}


//--------------------------------------------------------------------------------------------
// Deinterleave 4x4 block into RGBA channels and store each channel in seperate SSE registers
//--------------------------------------------------------------------------------------------

// Helper function to deinterleave the columns using high-low unpack logic to seperate into RGBA,
static inline void deinterleave_block(__m128i col0, __m128i col1, __m128i col2, __m128i col3,
                                      __m128i* r_plane, __m128i* g_plane, __m128i* b_plane, __m128i* a_plane) {
    
    // Interleave bytes from columns to group same channels together
    __m128i rg_01 = _mm_unpacklo_epi8(col0, col1);  // R0G0R1G1 R4G4R5G5 R8G8R9G9 R12G12R13G13
    __m128i ba_01 = _mm_unpackhi_epi8(col0, col1);  // B0A0B1A1 B4A4B5A5 B8A8B9A9 B12A12B13A13
    __m128i rg_23 = _mm_unpacklo_epi8(col2, col3);  // R2G2R3G3 R6G6R7G7 R10G10R11G11 R14G14R15G15
    __m128i ba_23 = _mm_unpackhi_epi8(col2, col3);  // B2A2B3A3 B6A6B7A7 B10A10B11A11 B14A14B15A15
    
    __m128i r_temp = _mm_unpacklo_epi8(rg_01, rg_23);  // R0R2R1R3 R4R6R5R7 R8R10R9R11 R12R14R13R15
    __m128i g_temp = _mm_unpackhi_epi8(rg_01, rg_23);  // G0G2G1G3 G4G6G5G7 G8G10G9G11 G12G14G13G15
    __m128i b_temp = _mm_unpacklo_epi8(ba_01, ba_23);  // B0B2B1B3 B4B6B5B7 B8B10B9B11 B12B14B13B15
    __m128i a_temp = _mm_unpackhi_epi8(ba_01, ba_23);  // A0A2A1A3 A4A6A5A7 A8A10A9A11 A12A14A13A15
    
    // Final shuffle to get proper order
    *r_plane = _mm_unpacklo_epi8(r_temp, _mm_unpackhi_epi8(r_temp, r_temp));
    *g_plane = _mm_unpacklo_epi8(g_temp, _mm_unpackhi_epi8(g_temp, g_temp));
    *b_plane = _mm_unpacklo_epi8(b_temp, _mm_unpackhi_epi8(b_temp, b_temp));
    *a_plane = _mm_unpacklo_epi8(a_temp, _mm_unpackhi_epi8(a_temp, a_temp));
}

// Helper macro to store the lower 32 bits of a 128-bit register
#define _mm_storeu_si32(p, a) _mm_store_ss((float*)(p), _mm_castsi128_ps(a))
static inline void store_planar_block(unsigned char* out_ptr, int width, int height, int x, int y,
                               __m128i r_plane, __m128i g_plane, __m128i b_plane, __m128i a_plane) {
    size_t plane_size = (size_t)width * height;

    for (int i = 0; i < 4; ++i) {
        // Calculate the offset for the current row in the 4x4 block
        size_t byte_offset = (size_t)(y + i) * width + x;

        // Store 4 bytes for each color plane
        _mm_storeu_si32(out_ptr + 0 * plane_size + byte_offset, r_plane);
        _mm_storeu_si32(out_ptr + 1 * plane_size + byte_offset, g_plane);
        _mm_storeu_si32(out_ptr + 2 * plane_size + byte_offset, b_plane);
        _mm_storeu_si32(out_ptr + 3 * plane_size + byte_offset, a_plane);

        // Shift the next 4 bytes of the plane into the lower 32 bits of the register
        r_plane = _mm_shuffle_epi32(r_plane, _MM_SHUFFLE(0, 3, 2, 1));
        g_plane = _mm_shuffle_epi32(g_plane, _MM_SHUFFLE(0, 3, 2, 1));
        b_plane = _mm_shuffle_epi32(b_plane, _MM_SHUFFLE(0, 3, 2, 1));
        a_plane = _mm_shuffle_epi32(a_plane, _MM_SHUFFLE(0, 3, 2, 1));
    }
}

unsigned char* deinterleave_rgb_block_sse(unsigned char* transposed_blocks, int width, int height) {
    unsigned char* planar_output = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if (!planar_output) {
        fprintf(stderr, "Failed to allocate planar buffer\n");
        return NULL;
    }

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            __m128i col0, col1, col2, col3;
            __m128i r_plane, g_plane, b_plane, a_plane;
            
            unsigned char* in_ptr = transposed_blocks + ((size_t)y * width + x) * CHANNELS_PER_PIXEL;

            load_transposed_block(in_ptr, &col0, &col1, &col2, &col3);
            deinterleave_block(col0, col1, col2, col3, &r_plane, &g_plane, &b_plane, &a_plane);
            store_planar_block(planar_output, width, height, x, y, r_plane, g_plane, b_plane, a_plane);
        }
    }
    return planar_output;
}


//---------------------------------------------------------------------------------
// Reinterleave 4x4 blocks of channels back into unified image and store rebuilt image back in memory
//---------------------------------------------------------------------------------

#define _mm_loadu_si32(p) _mm_castps_si128(_mm_load_ss((const float*)(p))) // Helper macro to load 32 bits into a 128-bit register

// Helper macro to load 32 bits into a 128-bit register
#define _mm_loadu_si32(p) _mm_castps_si128(_mm_load_ss((const float*)(p)))

static inline void load_planar_block(const unsigned char* in_ptr, int width, int height, int x, int y,
                                     __m128i* r_plane, __m128i* g_plane, __m128i* b_plane, __m128i* a_plane) {
    size_t plane_size = (size_t)width * height;

    // Read the 4 bytes for each of the 4 rows into temporary 32-bit integers
    int r_rows[4], g_rows[4], b_rows[4], a_rows[4];

    for (int i = 0; i < 4; ++i) {
        size_t byte_offset = (size_t)(y + i) * width + x;
        r_rows[i] = *(int*)(in_ptr + 0 * plane_size + byte_offset);
        g_rows[i] = *(int*)(in_ptr + 1 * plane_size + byte_offset);
        b_rows[i] = *(int*)(in_ptr + 2 * plane_size + byte_offset);
        a_rows[i] = *(int*)(in_ptr + 3 * plane_size + byte_offset);
    }

    // Insert each 32-bit integer into the correct lane of the 128-bit register
    *r_plane = _mm_set_epi32(r_rows[3], r_rows[2], r_rows[1], r_rows[0]);
    *g_plane = _mm_set_epi32(g_rows[3], g_rows[2], g_rows[1], g_rows[0]);
    *b_plane = _mm_set_epi32(b_rows[3], b_rows[2], b_rows[1], b_rows[0]);
    *a_plane = _mm_set_epi32(a_rows[3], a_rows[2], a_rows[1], a_rows[0]);
}

static inline void interleave_block(__m128i r_plane, __m128i g_plane, __m128i b_plane, __m128i a_plane,
                                    __m128i* col0, __m128i* col1, __m128i* col2, __m128i* col3) {
    __m128i tmp0, tmp1, tmp2, tmp3;
    tmp0 = _mm_unpacklo_epi16(r_plane, b_plane);
    tmp1 = _mm_unpackhi_epi16(r_plane, b_plane);
    tmp2 = _mm_unpacklo_epi16(g_plane, a_plane);
    tmp3 = _mm_unpackhi_epi16(g_plane, a_plane);

    *col0 = _mm_unpacklo_epi8(tmp0, tmp2);
    *col1 = _mm_unpackhi_epi8(tmp0, tmp2);
    *col2 = _mm_unpacklo_epi8(tmp1, tmp3);
    *col3 = _mm_unpackhi_epi8(tmp1, tmp3);
}

static inline void transpose_deinterleave_block( __m128i in0, __m128i in1, __m128i in2, __m128i in3, __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3) {
    __m128i tmp0 = _mm_unpacklo_epi32(in0, in1);
    __m128i tmp1 = _mm_unpacklo_epi32(in2, in3);
    __m128i tmp2 = _mm_unpackhi_epi32(in0, in1);
    __m128i tmp3 = _mm_unpackhi_epi32(in2, in3);
    *out0 = _mm_unpacklo_epi64(tmp0, tmp1);
    *out1 = _mm_unpackhi_epi64(tmp0, tmp1);
    *out2 = _mm_unpacklo_epi64(tmp2, tmp3);
    *out3 = _mm_unpackhi_epi64(tmp2, tmp3);
}

unsigned char* reinterleave_rgb_block_sse(unsigned char* input, int width, int height) {
    unsigned char* interleaved = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if (!interleaved) {
        fprintf(stderr, "Failed to allocate reinterleave buffer\n");
        return NULL;
    }

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += 4) {
            __m128i r0, r1, r2, r3;
            __m128i col0, col1, col2, col3;
            __m128i r_plane, g_plane, b_plane, a_plane;
            
            unsigned char* out_ptr = interleaved + ((size_t)y * width + x) * CHANNELS_PER_PIXEL;

            load_planar_block(input, width, height, x, y, &r_plane, &g_plane, &b_plane, &a_plane);
            interleave_block(r_plane, g_plane, b_plane, a_plane, &col0, &col1, &col2, &col3);
            transpose_deinterleave_block(col0, col1, col2, col3, &r0, &r1, &r2, &r3); // This is the inverse transpose
            store_interleaved_block(out_ptr, width, r0, r1, r2, r3);
        }
    }
    return interleaved;
}


//---------------------------------------------------------------------------------
// Reinterleave and store results AFTER GAUSSIAN FILTER OPERATIONS back to memory
//---------------------------------------------------------------------------------

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

static inline void interleave_and_store_rgba(unsigned char* output, __m128i packed_rgb, const unsigned char* input) {
    unsigned char* rgb_ptr = (unsigned char*)&packed_rgb;
    for (int i = 0; i < 4; i++) {
        output[i * 4 + 0] = rgb_ptr[i];
        output[i * 4 + 1] = rgb_ptr[i + 4];
        output[i * 4 + 2] = rgb_ptr[i + 8];
        output[i * 4 + 3] = input[i * 4 + 3];
    }
}
void store_rgba_results(unsigned char* output, __m128 red, __m128 green, __m128 blue, const unsigned char* input) {
    __m128i r_int, g_int, b_int;
    convert_float_int32(red, green, blue, &r_int, &g_int, &b_int);
    __m128i packed_rgb = pack_int8(r_int, g_int, b_int);
    interleave_and_store_rgba(output, packed_rgb, input);
}