#include "gaussian_filter.h"



/*---------------------------------------------MAIN----------------------------------------------------*/

int main(){

    // Iteratively benchmark each filtering technique and record the parameters (technique, sigma, kernel_size, cpu_time, wall_time) in an array
    int techniques[] = {1, 2, 3, 4}; // 1 = base, 2 = seperable, 3 = SSE_Base, 4 = SSE_Shuffle
    int n_techniques = sizeof(techniques) / sizeof(techniques[0]);
    int total_results = n_techniques * 8; // total # of iterations =  n_techniques*(max_sigma - starting_sigma)/step_size + 1 = n_techniques*(4 - 0.5)/0.5 + 1 = n_techniques*8
    
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
    
    for(int choice = 3; choice < n_techniques; choice++){ // For each filtering technique...
        filter_choice = techniques[choice];
        for(int i = 0; i <= n_sigma_steps; i++){

            float sigma = MIN_SIGMA + i * SIGMA_STEP; // Calculate the current sigma value
            int kernel_size = 2*(int)ceil(3*sigma) + 1; // First, calculate an acceptable kernel size given the sigma level
            BenchmarkResult result;  // Instantiate struct BenchmarkResult for holding processing times
            
            // Fill with initial parameters
            result.filter_choice = filter_choice;
            result.sigma = sigma;
            result.kernel_size = kernel_size;

            image_decode("test_1.png", kernel_size, sigma, filter_choice, &result); // Decode and initialize benchmark, update result
            results[result_idx++] = result; // fill array with updated results

            }
        }
    printf("All Tests Complete!\n");
    printf("-----\t-----\t-----\t--------\t---------\n");
    

    // Print the results into a table:
    print_statistics(results, total_results);

    free(results);
}