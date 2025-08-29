#include "../inc/debug.h"
#include "../inc/png_transform.h"
#include "../inc/image_operations.h"
#include "../inc/utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// --- Private Helper Functions for Debugging ---

static void debug_padding(unsigned char* image, int width, int height) {
    printf("Testing image_padding_transform...\n");
    PaddedImage* padded = image_padding_transform(image, width, height, 5);
    if (padded) {
        printf("Padding successful. Padded dimensions: %d x %d\n", padded->padded_width, padded->padded_height);
        save_image("debug_padding.png", padded->data, padded->padded_width, padded->padded_height, 1, 0);
        free(padded->data);
        free(padded);
    }
}

static void debug_transpose_base(unsigned char* image, int width, int height) {
    printf("Testing transpose_rgb_base...\n");
    unsigned char* transposed = transpose_rgb_base(image, width, height);
    if (transposed) {
        printf("Base transpose successful.\n");
        unsigned char* retransposed = transpose_rgb_base(transposed, height, width);
        save_image("debug_transpose_base.png", retransposed, width, height, 1, 0);
        free(transposed);
        free(retransposed);
    }
}

static void debug_transpose_sse(unsigned char* image, int width, int height) {
    printf("Testing transpose_image_sse reversibility...\n");

    PaddedImage* padded = image_padding_transform(image, width, height, 0);
    if (!padded) {
        fprintf(stderr, "Failed to pad image for SSE transpose test.\n");
        return;
    }

    unsigned char* transposed = transpose_rgb_block_sse(padded->data, padded->padded_width, padded->padded_height);
    if (transposed) {
        printf("SSE transpose successful.\n");
        
        unsigned char* retransposed_padded = retranspose_rgb_block_sse(transposed, padded->padded_width, padded->padded_height);
        
        unsigned char* final_image = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
        for(int y = 0; y < height; y++) {
            memcpy(final_image + (size_t)y * width * CHANNELS_PER_PIXEL,
                   retransposed_padded + (size_t)y * padded->padded_width * CHANNELS_PER_PIXEL,
                   (size_t)width * CHANNELS_PER_PIXEL);
        }

        save_image("debug_transpose_reversibility.png", final_image, width, height, 1, 0);
        free(transposed);
        free(retransposed_padded);
        free(final_image);
    }
    free(padded->data);
    free(padded);
}

static void debug_store_rgba(unsigned char* image, int width, int height) {
    printf("Testing store_rgba_results...\n");
    __m128 red = _mm_set_ps(255.0f, 0.0f, 0.0f, 255.0f);
    __m128 green = _mm_set_ps(0.0f, 255.0f, 0.0f, 255.0f);
    __m128 blue = _mm_set_ps(0.0f, 0.0f, 255.0f, 255.0f);

    unsigned char* output = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if(output){
        memcpy(output, image, (size_t)width * height * CHANNELS_PER_PIXEL);
        store_rgba_results(output, red, green, blue, image);
        printf("Store RGBA successful.\n");
        save_image("debug_store_rgba.png", output, width, height, 1, 0);
        free(output);
    }
}

static void save_planar_channel(const unsigned char* plane_data, int width, int height, const char* filename) {
    unsigned char* grayscale_image = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if (!grayscale_image) {
        fprintf(stderr, "Failed to allocate buffer for grayscale channel image.\n");
        return;
    }

    for (int i = 0; i < width * height; ++i) {
        unsigned char value = plane_data[i];
        grayscale_image[i * 4 + 0] = value;
        grayscale_image[i * 4 + 1] = value;
        grayscale_image[i * 4 + 2] = value;
        grayscale_image[i * 4 + 3] = 255;
    }

    save_image(filename, grayscale_image, width, height, 1, 0);
    free(grayscale_image);
}

static void debug_deinterleave_sse(unsigned char* image, int width, int height) {
        printf("Testing deinterleave/reinterleave process...\n");

    PaddedImage* padded = image_padding_transform(image, width, height, 0);
    if (!padded) {
        fprintf(stderr, "Failed to pad image for deinterleave test.\n");
        return;
    }
    
    unsigned char* transposed_blocks = transpose_rgb_block_sse(padded->data, padded->padded_width, padded->padded_height);
    unsigned char* planar_buffer = deinterleave_rgb_block_sse(transposed_blocks, padded->padded_width, padded->padded_height);

    if (!planar_buffer) {
        fprintf(stderr, "Deinterleaving failed.\n");
        free(padded->data);
        free(padded);
        free(transposed_blocks);
        return;
    }
    printf("Deinterleaving successful.\n");

    size_t plane_size = (size_t)padded->padded_width * padded->padded_height;
    save_planar_channel(planar_buffer, padded->padded_width, padded->padded_height, "debug_channel_red.png");
    save_planar_channel(planar_buffer + plane_size, padded->padded_width, padded->padded_height, "debug_channel_green.png");
    save_planar_channel(planar_buffer + 2 * plane_size, padded->padded_width, padded->padded_height, "debug_channel_blue.png");
    save_planar_channel(planar_buffer + 3 * plane_size, padded->padded_width, padded->padded_height, "debug_channel_alpha.png");
    printf("Saved grayscale visualizations for each channel.\n");

    free(planar_buffer);
    free(transposed_blocks);
    free(padded->data);
    free(padded);
}

static void debug_reinterleave_sse(unsigned char* image, int width, int height) {

        PaddedImage* padded = image_padding_transform(image, width, height, 0);
    if (!padded) {
        fprintf(stderr, "Failed to pad image for deinterleave test.\n");
        return;
    }
    
    unsigned char* transposed_blocks = transpose_rgb_block_sse(padded->data, padded->padded_width, padded->padded_height);
    unsigned char* planar_buffer = deinterleave_rgb_block_sse(transposed_blocks, padded->padded_width, padded->padded_height);


    unsigned char* reinterleaved_padded = reinterleave_rgb_block_sse(planar_buffer, padded->padded_width, padded->padded_height);
    if (!reinterleaved_padded) {
        fprintf(stderr, "Reinterleaving failed.\n");
        free(planar_buffer);
        free(transposed_blocks);
        free(padded->data);
        free(padded);
        return;
    }
    printf("Reinterleaving successful.\n");

    unsigned char* final_image = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    for(int y = 0; y < height; y++) {
        memcpy(final_image + (size_t)y * width * CHANNELS_PER_PIXEL,
               reinterleaved_padded + (size_t)y * padded->padded_width * CHANNELS_PER_PIXEL,
               (size_t)width * CHANNELS_PER_PIXEL);
    }
    save_image("debug_reinterleaved_image.png", final_image, width, height, 1, 0);
    printf("Saved reinterleaved image. Check if it matches the original.\n");

    free(planar_buffer);
    free(transposed_blocks);
    free(reinterleaved_padded);
    free(final_image);
    free(padded->data);
    free(padded);
}

void run_debug_mode() {
    int width, height;
    unsigned char* image = image_decode("test_1.png", &width, &height);
    if (!image) {
        fprintf(stderr, "Failed to decode image for debugging.\n");
        return;
    }

    int choice = 0;
    while (1) {
        printf("\n--- DEBUG MENU ---\n");
        printf("1. Test Image Padding (Returns image with padding to make its dimensions the closest multiple of 16) \n");
        printf("2. Test Base Transpose (Transpose x2 to get back original image) \n");
        printf("3. Test SSE Transpose (Transpose then Retranspose to get back original image since SSE is not symmetric) \n");
        printf("4. Test Store RGBA Results (Converts RGB values after SSE processing from float32 back to int8 via packing and maps back to output memory block) \n");
        printf("5. Test SSE Deinterleave (Returns three images comprising of the channels) \n");
        printf("6. Test SSE Reinterleave (Returns rebuilt image after deinterleaving) \n");
        printf("7. Exit \n");
        printf("Enter your choice: ");
        
        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n');
            choice = 0;
        }

        switch (choice) {
            case 1:
                debug_padding(image, width, height);
                break;
            case 2:
                debug_transpose_base(image, width, height);
                break;
            case 3:
                debug_transpose_sse(image, width, height);
                break;
            case 4:
                debug_store_rgba(image, width, height);
                break;
            case 5:
                debug_deinterleave_sse(image, width, height);
                break;
            case 6:
                debug_reinterleave_sse(image, width, height);
                break;
            case 7:
                free(image);
                printf("Exiting debug mode.\n");
                return;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}