#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#include "../inc/utility.h"
#include "../inc/gaussian_filter.h"
#include "gaussian_filter_cuda.h"
#include "../inc/lodepng/lodepng.h"

// Takes in data from Benchmark results and uses to compute summary statistics for each approach
void print_statistics(BenchmarkResult* results, int count) {
    double base_cpu_total = 0.0, base_wall_total = 0.0;
    double sep_cpu_total = 0.0, sep_wall_total = 0.0;
    double sse_cpu_total = 0.0, sse_wall_total = 0.0;
    double shuffle_cpu_total = 0.0, shuffle_wall_total = 0.0;
    double cuda_gpu_total = 0.0, cuda_wall_total = 0.0;
    int base_count = 0, sep_count = 0, sse_count = 0, shuffle_count = 0, cuda_count = 0;


    // Print total times for each filter_choice
    for (int i = 0; i < count; i++) {
        switch(results[i].filter_choice) {
            case 1: // Base
                base_cpu_total += results[i].cpu_time;
                base_wall_total += results[i].wall_time;
                base_count++;
                break;
            case 2: // Separable
                sep_cpu_total += results[i].cpu_time;
                sep_wall_total += results[i].wall_time;
                sep_count++;
                break;
            case 3: // SSE Base
                sse_cpu_total += results[i].cpu_time;
                sse_wall_total += results[i].wall_time;
                sse_count++;
                break;
            case 4: // SSE Shuffle
                shuffle_cpu_total += results[i].cpu_time;
                shuffle_wall_total += results[i].wall_time;
                shuffle_count++;
                break;
            case 5: // Base CUDA
                cuda_gpu_total += results[i].cpu_time;
                cuda_wall_total += results[i].wall_time;
                cuda_count++;
                break;                
        }
    }

    // Print average times for each filter_choice
    printf("\nAverage Times:\n");
    if (base_count > 0)
        printf("Base:     CPU: %.3fms, Wall: %.3fms\n", 
               (base_cpu_total/base_count)*1000, (base_wall_total/base_count)*1000);
    if (sep_count > 0)
        printf("Separable: CPU: %.3fms, Wall: %.3fms\n", 
               (sep_cpu_total/sep_count)*1000, (sep_wall_total/sep_count)*1000);
    if (sse_count > 0)
        printf("SSE Base:  CPU: %.3fms, Wall: %.3fms\n", 
               (sse_cpu_total/sse_count)*1000, (sse_wall_total/sse_count)*1000);
    if (shuffle_count > 0)
        printf("SSE Shuffle: CPU: %.3fms, Wall: %.3fms\n", 
               (shuffle_cpu_total/shuffle_count)*1000, (shuffle_wall_total/shuffle_count)*1000);
    if (cuda_count > 0)
        printf("CUDA Base: CPU: %.3fms, Wall: %.3fms\n", 
               (cuda_gpu_total/cuda_count)*1000, (cuda_wall_total/cuda_count)*1000);
}


/// Run and benchmark filters, measure wall time (absolute time) and CPU time (computational time) needed to finish applying the gaussian filter to the image
void measure_filter_time(unsigned char* image, int width, int height, float sigma, int kernel_size, int filter_choice, BenchmarkResult *result) {

    clock_t start_cpu, end_cpu;
    time_t start_wall, end_wall;
    double cpu_time_used, wall_time_used;
    
    printf("\n=== Processing Image ===\n");
    printf("Image size: %dx%d pixels\n", width, height);
    printf("Kernel size: %d\n", kernel_size);
    printf("Sigma: %.2f\n", sigma);

    start_cpu = clock();
    start_wall = time(NULL);
    
    if (filter_choice == 1) {
        printf("Using 2D Gaussian Filter (Base)...\n");
        gaussian_filter_base(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 2) {
        printf("Using Separable Gaussian Filter...\n");
        gaussian_filter_separable(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 3) {
        printf("Using Base SSE Sep. Gaussian Filter (Base SSE)...\n");
        gaussian_filter_sse_base(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 4) {
        printf("Using SSE Load Shuffle Sep. Gaussian Filter...\n");
        gaussian_filter_sse_shuffle(image, width, height, sigma, kernel_size);
    } else if (filter_choice == 5) {
        printf("Using Base CUDA Sep. Gaussian Filter (Base Cuda)...\n");
        gaussian_filter_cuda(image, width, height, sigma, kernel_size);
    }
    else {
        fprintf(stderr, "Invalid filter choice: %d\n", filter_choice);
        return;
    }
    
    end_cpu = clock();
    end_wall = time(NULL);
    
    cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    wall_time_used = difftime(end_wall, start_wall);

    result->cpu_time = cpu_time_used;
    result->wall_time = wall_time_used;
    
    printf("CPU time: %.4f seconds\n", cpu_time_used);
    printf("Wall time: %.4f seconds\n", wall_time_used);
    printf("=== Processing Complete ===\n\n");
}

void countdown(int seconds){
    time_t start_time = time(NULL); // get start time
    time_t current_time;
    int remaining_time = seconds;

    while(remaining_time > 0){
        printf("\r%d\n ", remaining_time);
        do{
            current_time = time(NULL); // repeatedly update current_time and wait...
        } while(current_time == start_time); // until current_time != start_time and...
        
        start_time = current_time; // update start time and wait once more until current time again no longer equals start
        remaining_time--; // repeat until remaining time goes down to 0
    }
    printf("\n----------Starting Test----------\n");
}

// Saves an image to a file.
int save_image(const char* filename, const unsigned char* image_data, 
               int width, int height, int filter_choice, int kernel_size) {
    // Output filter names
    const char* filter_names[] = {"base", "separable", "sse_base", "sse_shuffle", "cuda"};

    // Ensure output directory exists
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        mkdir("output", 0700);
    }

    // Truncate the extension (i.e. ".png")
    const char *dot = strrchr(filename, '.');
    size_t basename_len = dot ? (size_t)(dot - filename) : strlen(filename);

    // Compose output filename: output/<filename>_k<kernel_size>_<filter>.png
    char output_filename[2048];
    snprintf(output_filename, sizeof(output_filename),
            "output/%.*s_k%d_%s.png",
            (int)basename_len, filename, kernel_size, filter_names[filter_choice - 1]);

    // Save the image (standard lodepng method)
    printf("Attempting to save to: %s\n", output_filename);
    unsigned error = lodepng_encode32_file(output_filename, image_data, width, height);
    if(error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return -1;
    }

    printf("Saved processed image as: %s\n", output_filename);
    return 0;
}