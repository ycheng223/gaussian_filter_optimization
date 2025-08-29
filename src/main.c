#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include "../inc/gaussian_filter.h"
#include "../inc/png_transform.h"
#include "../inc/image_operations.h"
#include "../inc/utility.h"
#include "../inc/debug.h"

/*---------------------------------------------MAIN----------------------------------------------------*/

int main(int argc, char **argv){


    // Check for the "debug" command-line argument
    if (argc > 1 && strcmp(argv[1], "debug") == 0) {

    // If present, run the interactive debug menu
        run_debug_mode();
    } else {

        // Otherwise, iteratively benchmark each filtering technique and record the parameters (technique, sigma, kernel_size, cpu_time, wall_time) in an array
        int techniques[] = {1, 2, 3, 4}; // 1 = base, 2 = seperable, 3 = SSE_Base, 4 = SSE_Shuffle
        int n_techniques = sizeof(techniques) / sizeof(techniques[0]);
        int total_results = n_techniques * 8; // total # of iterations =  n_techniques*(max_sigma - starting_sigma)/step_size + 1 = n_techniques*8
        
        BenchmarkResult *results = malloc(total_results * sizeof(BenchmarkResult)); // Total # iterations * size of struct = total bytes of memory we need to allocate
        if(!results){
            fprintf(stderr, "Memory Allocation Failed!\n");
            return 1;
        }

        printf("\nThis program will compare the computational efficiency of different\n");
        printf("Gaussian filtering techniques by applying successively larger Gaussian\n");
        printf("kernels onto a test image and measuring the computational time\n");
        printf("\n\n\nPress ENTER to begin...\n");
        getchar();

        printf("\nStarting test in.....\n");
        countdown(3);

        int n_sigma_steps = (int)((MAX_SIGMA - MIN_SIGMA) / SIGMA_STEP); // given min, max, and step size of sigma, get total number of times we run the benchmark
        int filter_choice;
        int result_idx = 0;
        
        for(int choice = STARTING_FILTER; choice < n_techniques; choice++){ // For each filtering technique...
            filter_choice = techniques[choice];
            for(int i = 0; i <= n_sigma_steps; i++){

                float sigma = MIN_SIGMA + i * SIGMA_STEP; // Calculate the current sigma value
                int kernel_size = 2*(int)ceil(3*sigma) + 1; // First, calculate an acceptable kernel size given the sigma level
                BenchmarkResult result;  // Instantiate struct BenchmarkResult for holding processing times
                
                // Fill with initial parameters
                result.filter_choice = filter_choice;
                result.sigma = sigma;
                result.kernel_size = kernel_size;

                unsigned char* image = image_decode("test_1.png", &result.width, &result.height); // Decode png image to 1D RGB on heap
                if (!image) {
                    fprintf(stderr, "Failed to decode image.\n");
                    free(results);
                    return 1;
                }

                // Measure filter time
                measure_filter_time(image, result.width, result.height, kernel_size, sigma, filter_choice, &result);

                // Encode the image to a memory buffer
                save_image("test_1.png", image, result.width, result.height, filter_choice, kernel_size);

                // Free the image after encoding
                free(image);

                // Store the result
                results[result_idx++] = result;

                }
            }
        printf("All Tests Complete!\n");
        printf("-----\t-----\t-----\t--------\t---------\n");
        

        // Print the results into a table:
        print_statistics(results, total_results);

        free(results);
    }
}