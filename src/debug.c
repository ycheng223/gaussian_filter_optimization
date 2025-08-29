#include "../inc/debug.h"
#include "../inc/png_transform.h"
#include "../inc/image_operations.h"
#include "../inc/utility.h"

#include <stdio.h>
#include <stdlib.h>

// --- Private Helper Functions for Debugging ---

static void debug_padding(unsigned char* image, int width, int height) {
    printf("Testing image_padding_transform...\n");
    PaddedImage* padded = image_padding_transform(image, width, height, 5); // Using a kernel size of 11 (range 5) for testing
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
        // To visualize, we need to transpose it back
        unsigned char* retransposed = transpose_rgb_base(transposed, height, width);
        save_image("debug_transpose_base.png", retransposed, width, height, 1, 0);
        free(transposed);
        free(retransposed);
    }
}

static void debug_transpose_sse(unsigned char* image, int width, int height) {
    printf("Testing transpose_rgb_block_sse...\n");
    unsigned char* transposed = transpose_rgb_block_sse(image, width, height);
    if (transposed) {
        printf("SSE transpose successful.\n");
        // To visualize, we need to transpose it back
        unsigned char* retransposed = transpose_rgb_block_sse(transposed, height, width);
        save_image("debug_transpose_sse.png", retransposed, width, height, 1, 0);
        free(transposed);
        free(retransposed);
    }
}

static void debug_store_rgba(unsigned char* image, int width, int height) {
    printf("Testing store_rgba_results...\n");
    // Create some dummy data to store
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


// --- Public Function to Run the Debug Menu ---

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
        printf("1. Test Image Padding\n");
        printf("2. Test Base Transpose\n");
        printf("3. Test SSE Transpose\n");
        printf("4. Test Store RGBA Results\n");
        printf("5. Exit\n");
        printf("Enter your choice: ");
        
        if (scanf("%d", &choice) != 1) {
            // Clear invalid input
            while (getchar() != '\n');
            choice = 0; // Reset choice
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
                free(image);
                printf("Exiting debug mode.\n");
                return;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    }
}
