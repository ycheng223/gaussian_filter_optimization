#include "gaussian_filter.h"


// Decode the image into 1D array of rgb values and load into a contiguous memory space
void image_decode(const char* filename, int kernel_size, float sigma, int filter_choice, BenchmarkResult *result) {

    unsigned error;
    unsigned char* image = 0;
    int width, height;
    char input_path[256];
    char output_path[256];
    
    // Construct absolute paths to input and output folders
    snprintf(input_path, sizeof(input_path), "./input/%s", filename);
    snprintf(output_path, sizeof(output_path), "./output/%s", filename);

    error = lodepng_decode24_file(&image, (unsigned*)&width, (unsigned*)&height, input_path); // decode and store image contiguously into memory as 1D array, 3 channels per pixel (i.e. RGB, NO ALPHA)
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //note: to include alpha, we would the the lodepng_decode32_file function

    else{
        measure_filter_time(image, width, height, sigma, kernel_size, filter_choice, result); // measure time and apply selected gaussian filter
        image_encode(output_path, image, width, height); // re-encode as image and save to disc in "output" folder
    }

    free(image);

    image = NULL;
}


// Encode array of RGB values back into image to get back the tranformed image after the gaussian filter is applied
void image_encode(const char* filepath, const unsigned char* image, int width, int height) {
 
    unsigned error = lodepng_encode24_file(filepath, image, (unsigned)width, (unsigned)height); //encode and save to disc as "filename"
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

/// Measure wall time (absolute time) and CPU time (computational time) needed to finish applying the gaussian filter to the image
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


// Helper function to deal with the edges of image -> if kernel extends beyond the edges 
// of the image, fill it with the pixel value of the nearest edge and update the image data
int border_clamp(int width, int height, int x, int y){
    int clamped_x = x;
    int clamped_y = y;

    if(clamped_x < 0) clamped_x = 0;
    if(clamped_x >= width) clamped_x = width - 1;

    if(clamped_y < 0) clamped_y = 0;
    if(clamped_y >= height) clamped_y = height - 1;

    return clamped_y * width + clamped_x; // Returns index of the nearest edge pixel for a 1D row-major array
}


void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size){

    // We are replacing each individual pixel in the image with the weighted average of it and its neighboring pixels in a nxn matrix (gaussian kernel) where n = kernel_size
    // Kernel weights are normally calculated by applying eqn for gaussian distribution to each coordinate on the gaussian kernel relative to it's center.
    // i.e. { [e^-(x^2 + y^2)] / (2*sigma^2) } where {x,y} = [0,0], [0,1], [1,0], [1,1] .... [n/2,n/2] -> note that it is n/2 because the distance is relative to the center of the kernel.
    // Get the weighted average by by multiplying the RGB value of each pixel overlayed by the gaussian kernel (i.e. dot product) and calculating their weighted average (i.e. (sum of dot products)/(weighted_sum))
    // The greater the blur (i.e. variance/sigma) the larger the gaussian kernel needs to be to maintain precision but more on that later...

    int range = kernel_size / 2;
    // First we allocate a temp buffer for the convolution
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    for (int y = 0; y < height; y++) { // For every row...
        for (int x = 0; x < width; x++) { // For every pixel...
            for (int c = 0; c < 3; c++) { // For every RGB channel... (this is basically every value in the 1D image array)

                float sum = 0.0f; // Initialize a weighted sum used to calculate weighted average
                float weight_sum = 0.0f; // Initialize a sum of weights to calculate weighted average


                // Now we iterate through every element in the n x n kernel and calculate it's weight based on gaussian distribution
                // For each weight, multiply by it's corresponding pixel value in the image data and use these values to update
                // The weighted sum and sum of weights for this gaussian kernel
                for (int kernel_y = -range; kernel_y <= range; kernel_y++) { // for every row in the kernel...
                    for (int kernel_x = -range; kernel_x <= range; kernel_x++) { //for every column in the kernel...
                        int x_neighbor = x + kernel_x; // absolute x position of neighboring pixel
                        int y_neighbor = y + kernel_y; // absolute y position of neighboring pixel
                        int clamped_index = border_clamp(width, height, x_neighbor, y_neighbor); // clamp pixel index to index of nearest edge if kernel position exceeds edge 
                        float pixel_value = image[3 * clamped_index + c]; // start of pixel's RGB values in the 1D array of image data stored in memory (hence, 3 * clamped_index + c)
                        float weight = expf(-(kernel_x * kernel_x + kernel_y * kernel_y) / (2 * sigma * sigma)); // calculate gaussian weights ie. [e^(x^2 + y^2)] / (2 * sigma^2)
                        sum += pixel_value * weight; // iteratively sum dot products of pixel_rgb and gaussian weights to get numerator of weighted avg
                        weight_sum += weight; // iteratively sum gaussian weights to get the denominator
                    }

                }
                temp[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); // weighted average = (sum of dot products) / sum of gaussian weights), clamp between 0 and 255
            }
        }
    }
    memcpy(image, temp, width * height * 3); // copy contents of temp buffer (i.e. blurred pixels) back to image buffer
    free(temp); // release memory allocated for temp buffer
}


void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size){
    int range = kernel_size / 2;
    
    // Next to cut down on computational cost, we will transform the 2D Gaussian filter into the product of
    // 2 1D Gaussian filters. This is possible because gaussian processes are separable and hence, any matrix
    // can be represented as the product of two 1D filters.

    // First we allocate a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(width * height * CHANNELS_PER_PIXEL); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Horizontal pass: apply 1D Gaussian kernel to each row
    for(int y = 0; y < height; y++){ // For every row...
        for(int x = 0; x < width; x++){ // For every pixel....
            for(int c = 0; c < 3; c++){ //For every RGB channel....
                float sum = 0.0f; // weighted sum to calculate blur for pixel in center of kernel window
                float weight_sum = 0.0f; // Sum of weights for normalization -> calculate weighted average
                for (int kernel_index = -range; kernel_index <= range; kernel_index++){ //For each relative pixel position (to center of kernel) in the kernel window...
                    int x_neighbor = x + kernel_index; // get the absolute index of the pixel
                    int clamped_index = border_clamp(width, height, x_neighbor, y); //clamp x_neighbor if kernel extends beyond edge of image
                    float pixel_value = image[3 * clamped_index + c]; //Get neighbor pixel value for current channel
                    float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma)); //Calculate 1D Gaussian weight
                    sum += pixel_value * weight; //Multiply and accumulate
                    weight_sum += weight; //Accumulate weights for normalization
                }
                temp[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); //Normalize and store result
            }
        }
    }
    
    // Vertical pass: apply 1D Gaussian kernel to each column of the horizontally filtered result
    for(int y = 0; y < height; y++){ // For every row...
        for(int x = 0; x < width; x++){ // And every pixel....
            for(int c = 0; c < 3; c++){ //And every RGB channel....
                float sum = 0.0f; // weighted sum to calculate blur for pixel in center of kernel window
                float weight_sum = 0.0f; // Sum of weights for normalization -> calculate weighted average
                for (int kernel_index = -range; kernel_index <= range; kernel_index++){ //For each pixel index in the kernel window...
                    int y_neighbor = y + kernel_index; // get the absolute index of the pixel
                    int clamped_index = border_clamp(width, height, x, y_neighbor); //clamp y_neighbor if kernel extends beyond edge of image
                    float pixel_value = temp[3 * clamped_index + c]; //Get neighbor pixel value from temp buffer (horizontally filtered)
                    float weight = expf(-(kernel_index * kernel_index) / (2 * sigma * sigma)); //Calculate 1D Gaussian weight (same as horizontal)
                    sum += pixel_value * weight; //Multiply and accumulate
                    weight_sum += weight; //Accumulate weights for normalization
                }
                image[3 * (y * width + x) + c] = (unsigned char)fminf(fmaxf(sum / weight_sum, 0.0f), 255.0f); //Normalize and store result back to image
            }
        }
    }
    
    free(temp); // release memory allocated for temp buffer
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


/*---------------------------------------------MAIN----------------------------------------------------*/

int main(){

    // Iteratively benchmark each filtering technique and record the parameters (technique, sigma, kernel_size, cpu_time, wall_time) in an array
    int techniques[] = {1, 2}; // 1 = base, 2 = seperable
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

    float sigma = 0.5f;
    int filter_choice;
    int result_idx = 0;
    for(int choice = 0; choice < n_techniques; choice++){ // For each filtering technique...
        filter_choice = techniques[choice];
        sigma = 0.5f; //reset sigma when we benchmark a new technique
        while(sigma <= MAX_SIGMA){ // Up to a "large" sigma, perform the benchmark on incremental sigma levels and log the results
            int kernel_size = 2*(int)ceil(3*sigma) + 1; // First, calculate an acceptable kernel size given the sigma level
            BenchmarkResult result;  // Instantiate struct BenchmarkResult for holding processing times
            
            // Fill with initial parameters
            result.filter_choice = filter_choice;
            result.sigma = sigma;
            result.kernel_size = kernel_size;

            image_decode("test_1.png", kernel_size, sigma, filter_choice, &result); // Decode and initialize benchmark, update result
            results[result_idx++] = result; //fill array with updated results

            sigma += 0.5; // Increment sigma linearly in half steps
            }
        }
    printf("All Tests Complete!\n");
    printf("-----\t-----\t-----\t--------\t---------\n");
    

    // Print the results into a table:
    printf("\n=== Benchmark Results ===\n");
    printf("%-10s %-8s %-8s %-12s %-12s\n", "Filter", "Kernel", "Sigma", "CPU Time", "Wall Time");
    printf("%-10s %-8s %-8s %-12s %-12s\n", "------", "------", "-----", "--------", "----------");

    for (int i = 0; i < result_idx; i++) {
        const char* filter_name = (results[i].filter_choice == 1) ? "Base" : "Separable";
        printf("%-10s %-8d %-8.1f %-12.4f %-12.4f\n",
            filter_name,
            results[i].kernel_size,
            results[i].sigma,
            results[i].cpu_time,
            results[i].wall_time
        );
    }

    // Calculate total and average computational times for each Gaussian filtering technique
    double base_cpu_total = 0, base_wall_total = 0;
    double sep_cpu_total = 0, sep_wall_total = 0;
    int base_count = 0, sep_count = 0;

    for (int i = 0; i < result_idx; i++) {
        if (results[i].filter_choice == 1) {
            base_cpu_total += results[i].cpu_time;
            base_wall_total += results[i].wall_time;
            base_count++;
        } else {
            sep_cpu_total += results[i].cpu_time;
            sep_wall_total += results[i].wall_time;
            sep_count++;
        }
    }

    printf("\n=== Total Times ===\n");
    printf("Base Filter - Total CPU: %.4f, Total Wall: %.4f\n", 
        base_cpu_total, base_wall_total);
    printf("Separable Filter - Total CPU: %.4f, TOtal Wall: %.4f\n", 
        sep_cpu_total, sep_wall_total);


    printf("\n=== Average Times ===\n");
    printf("Base Filter - Average CPU: %.4f, Average Wall: %.4f\n", 
        base_cpu_total / base_count, base_wall_total / base_count);
    printf("Separable Filter - Average CPU: %.4f, Average Wall: %.4f\n", 
        sep_cpu_total / sep_count, sep_wall_total / sep_count);
    printf("\nTotal tests completed: %d\n", result_idx);
    free(results);
}