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

static void save_planar_channel(const unsigned char* plane_data, int width, int height, const char* filename, int channel_index) {
    unsigned char* channel_image = (unsigned char*)malloc((size_t)width * height * CHANNELS_PER_PIXEL);
    if (!channel_image) {
        fprintf(stderr, "Failed to allocate buffer for channel image.\n");
        return;
    }

    for (int i = 0; i < width * height; ++i) {
        unsigned char value = plane_data[i];
        // Set the red channel to the intensity value, green and blue to 0
        channel_image[i * 4 + 0] = (channel_index == 0) ? value : 0; // Red
        // Set the green channel to the intensity value, red and blue to 0
        channel_image[i * 4 + 1] = (channel_index == 1) ? value : 0; // Green
        // Set the blue channel to the intensity value, red and green to 0
        channel_image[i * 4 + 2] = (channel_index == 2) ? value : 0; // Blue
        channel_image[i * 4 + 3] = 255;                              // Alpha
    }

    save_image(filename, channel_image, width, height, 1, 0);
    free(channel_image);
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

static void debug_deinterleave_base(unsigned char* image, int width, int height) {
    printf("Testing deinterleave_rgb_base...\n");

    unsigned char* deinterleaved = deinterleave_rgb_base(image, width, height);
    if (!deinterleaved) {
        fprintf(stderr, "Base deinterleaving failed.\n");
        return;
    }
    printf("Base deinterleaving successful.\n");

    size_t plane_size = (size_t)width * height;
    save_planar_channel(deinterleaved, width, height, "debug_channel_base_red.png", 0);
    save_planar_channel(deinterleaved + plane_size, width, height, "debug_channel_base_green.png", 1);
    save_planar_channel(deinterleaved + 2 * plane_size, width, height, "debug_channel_base_blue.png", 2);
    save_planar_channel(deinterleaved + 3 * plane_size, width, height, "debug_channel_base_alpha.png", 3);
    printf("Saved grayscale visualizations for each channel.\n");

    free(deinterleaved);
}

static void debug_reinterleave_base(unsigned char* image, int width, int height) {
    printf("Testing reinterleave_rgb_base...\n");

    // First, deinterleave the image using the base function.
    unsigned char* deinterleaved = deinterleave_rgb_base(image, width, height);
    if (!deinterleaved) {
        fprintf(stderr, "Base deinterleaving failed during reinterleave test.\n");
        return;
    }

    // Now, reinterleave the deinterleaved data.
    unsigned char* reinterleaved = reinterleave_rgb_base(deinterleaved, width, height);
    if (!reinterleaved) {
        fprintf(stderr, "Base reinterleaving failed.\n");
        free(deinterleaved);
        return;
    }
    printf("Base reinterleaving successful.\n");

    // Save the final image to check if it matches the original.
    save_image("debug_reinterleaved_base_image.png", reinterleaved, width, height, 1, 0);
    printf("Saved reinterleaved base image. Check if it matches the original.\n");

    free(deinterleaved);
    free(reinterleaved);
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
        printf("3. Test Base Deinterleave (Returns 3 deinterleaved images by channel) \n");
        printf("4. Test Base Reinterleave (deinterleaves image the recombines channels to reconstruct -> return original image) \n");
        printf("5. Exit \n");
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
                debug_deinterleave_base(image, width, height);
                break;
            case 4:
                debug_reinterleave_base(image, width, height);
                break;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}