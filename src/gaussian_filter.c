#include "../inc/gaussian_filter.h"
#include "../inc/gaussian_processing.h"
#include "../inc/image_operations.h"
#include "../inc/utility.h"
#include <string.h>


void gaussian_filter_base(unsigned char* image, int width, int height, float sigma, int kernel_size){

    // We are replacing each individual pixel in the image with the weighted average of it and its neighboring pixels in a nxn matrix (gaussian kernel) where n = kernel_size
    // Kernel weights are normally calculated by applying eqn for gaussian distribution to each coordinate on the gaussian kernel relative to it's center.
    // i.e. { [e^-(x^2 + y^2)] / (2*sigma^2) } where {x,y} = [0,0], [0,1], [1,0], [1,1] .... [n/2,n/2] -> note that it is n/2 because the distance is relative to the center of the kernel.
    // Get the weighted average by by multiplying the RGB value of each pixel overlayed by the gaussian kernel (i.e. dot product) and calculating their weighted average (i.e. (sum of dot products)/(weighted_sum))
    // The greater the blur (i.e. variance/sigma) the larger the gaussian kernel needs to be to maintain precision but more on that later...

    int range = kernel_size / 2;
    // First we allocate a temp buffer for the convolution
    unsigned char* temp = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height)); //total dimension will be width * height * 4 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_kernel_base(image, temp, x, y, width, height, sigma, range);
        }
    }

    memcpy(image, temp, PADDED_IMG_SIZE(width, height));
    free(temp);
}




// Next to cut down on computational cost, we will transform the 2D Gaussian filter into the product of
// 2 1D Gaussian filters. This is possible because gaussian processes are separable and hence, any matrix
// can be represented as the product of two 1D filters.
void gaussian_filter_separable(unsigned char* image, int width, int height, float sigma, int kernel_size) {
    int range = kernel_size / 2;

    // First we allocate a temp buffer for the horizontal pass (i.e. convolve the rows with the normalized 1D gaussian kernel calculated above)
    unsigned char* temp = (unsigned char*)malloc(PADDED_IMG_SIZE(width, height)); //total dimension will be width * height * 3 channels (i.e. RGB)
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Horizontal pass (is_vertical = 0)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_separable_kernel(image, temp, x, y, width, height, sigma, range, 0);
        }
    }
    // Vertical pass (is_vertical = 1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            process_separable_kernel(temp, image, x, y, width, height, sigma, range, 1);
        }
    }
    free(temp); // release memory allocated for temp buffer
}