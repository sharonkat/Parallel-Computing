%%writefile convolution.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lodepng.h"
#include "wm.h"
#include <time.h>
# define MASK_SIZE 3

__global__ void convolution(unsigned char *new_image, unsigned char *old_image, int bits_per_block, int new_width, int new_size, int old_width, float* mask) {

	double weight_val, input_val, sum = 0;
	long int input_row, input_col;
    const int blockIndex = blockIdx.x * bits_per_block;

	for (int k = blockIndex; k < blockIndex + bits_per_block && k < new_size; k++) {
        const int i = k / new_width;
        const int j = k % new_width;
        
        // Skip ALPHA channel
		for (int rgb = 0; rgb < 3; rgb++) {
			sum = 0;
			for (int ii = 0; ii < MASK_SIZE; ii++) {
				for (int jj = 0; jj < MASK_SIZE; jj++) {
					input_row = i + ii;
					input_col = j + jj;
					input_val = old_image[4 * old_width * input_row + 4 * input_col + rgb];
                    weight_val =  mask[ii * MASK_SIZE + jj];

					sum += (float)round(input_val * weight_val);
				}
			}

            if (sum < 0) {
                sum = 0;
            } else if (sum > 255) {
                sum = 255;
            }
			new_image[4*new_width*i + 4*j + rgb] = sum;
		}
		new_image[4*new_width*i + 4*j + 3] = old_image[4*old_width*i + 4*j + 3];
	}
}

void process(char* input_filename, char* output_filename, int num_threads) {

    unsigned char *image;
    unsigned char *d_old_image, *d_convolved;
    unsigned width, height;
    int size;
    long int new_size;
    float *mask;

    unsigned error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return;
    }

    size = width*height;    // Size of one "tile" (r, g, b, or a)
    const int new_width = width - 2;
    const int new_height = height - 2;
    new_size = new_width*new_height;

    // Allocate space on GPU
    cudaMallocManaged((void**)&d_old_image, size*4*sizeof(unsigned char));
    cudaMallocManaged((void**)&d_convolved, new_size*4*sizeof(unsigned char));

    cudaMemcpy(d_old_image, image, size*4*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Allocate space for mask
    cudaMalloc((void**)&mask, MASK_SIZE*MASK_SIZE*sizeof(float));
    cudaMemcpy(mask, w, MASK_SIZE*MASK_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    const int bits_per_block = (new_size*4 + num_threads - 1) / num_threads;

    // Call Kernel
    convolution<<<num_threads, 1>>>(d_convolved, d_old_image, bits_per_block, new_width, new_size, width, mask);
    cudaDeviceSynchronize();

    lodepng_encode32_file(output_filename, d_convolved, new_width, new_height);

    cudaFree(d_old_image);
    cudaFree(d_convolved);
}

int main(int argc, char* argv[]) {

	if (argc != 4) {
		printf("Please include all variables.\n");
		return 1;
	}
	char* input_image = argv[1];
	char* output_image = argv[2];
	char* t;
	long num_threads = strtol(argv[3], &t, 10);

	clock_t start_time = clock();
	process(input_image, output_image, num_threads);

	clock_t end_time = clock();
	long total_time = (end_time - start_time);
    printf("Time recorded for %ld threads: %ld.\n", num_threads, total_time);

	return 0;
}