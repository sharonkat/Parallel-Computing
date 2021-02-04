%%writefile drum.cu


#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//Constants
#define G 0.75
#define p 0.5
#define n 0.0002




__global__ void finiteDrumUpdate(float *d_u, float *d_u1, float *d_u2, int arraySize, int size, int numBlocks, int threadsPerBlock)
{
    int id = threadIdx.x * numBlocks + blockIdx.x;
    for (int i = id; i < arraySize; i += threadsPerBlock * numBlocks)
    {
        //Update all the elements depending on their position in the matrix
     
        // first the corners
        if (i == 0 || i == size - 1 || i == (size - 1) * size || i == (arraySize - 1)) {
            d_u[i] = p * (d_u1[i + 1] + d_u1[i + 2 * size + 1] + d_u1[i + size] + d_u1[i + size + 2] - 4 * d_u1[i + size + 1]);
            d_u[i] += (2 * d_u1[i + size + 1] - (1 - n) * d_u2[i + size + 1]);
            d_u[i] *= 2*G/(1+n);
        // the sides in-between corners     
        } else if (i < size) { // Top side
            d_u[i] =  p * (d_u1[i] + d_u1[i + 2 * size] + d_u1[i + size - 1] + d_u1[i + size + 1] - 4 * d_u1[i + size]);
            d_u[i] += 2 * d_u1[i + size] - (1 - n) * d_u2[i + size];
            d_u[i] *= G/(1+n);
        }else if (i % size == 0) { // Left side
            d_u[i] = p * (d_u1[i + 1 - size] + d_u1[i + 1 + size] + d_u1[i] + d_u1[i + 2] - 4 * d_u1[i + 1]);
            d_u[i] += (2 * d_u1[i + 1] - (1 - n) * d_u2[i + 1]);
            d_u[i] *= G/(1+n);
        } else if (i % size == size - 1) { // Right side
            d_u[i] = p * (d_u1[i - 1 - size] + d_u1[i - 1 + size] + d_u1[i - 2] + d_u1[i] - 4 * d_u1[i - 1]);
            d_u[i] += 2 * d_u1[i - 1] - (1 - n) * d_u2[i - 1];
            d_u[i] *= G/(1+n);
        }else if (i > size * size - size) { // Bottom side
            d_u[i] = p * (d_u1[i - 2 * size] + d_u1[i] + d_u1[i - size - 1] + d_u1[i - size + 1] - 4 * d_u1[i - size]);
            d_u[i] += 2 * d_u1[i - size] - (1 - n) * d_u2[i - size];
            d_u[i] *= G/(1+n);
        }
        
        // and finally the middle elements
        else
        {
            d_u[i] = p * (d_u1[i - size] + d_u1[i + size] + d_u1[i - 1] + d_u1[i + 1] - 4 * d_u1[i]);
            d_u[i] += 2 * d_u1[i] - (1 - n) * d_u2[i];
            d_u[i] /= (1+n);
        }
     
        if (i == (arraySize / 2 + size / 2))
        {
            printf("\n(%d, %d): %.6f", size / 2, size / 2, d_u[i]);
        }
    }
}

void process(int size, int numIterations, int numBlocks, int threadsPerBlock)
{
    int arraySize = size * size;
    int arrayMemSize = arraySize * sizeof(float);
    float *u = (float *)malloc(arrayMemSize);
    float *u1 = (float *)malloc(arrayMemSize);
    float *u2 = (float *)malloc(arrayMemSize);
    
    // set all values in array to 0
    for (int i = 0; i < arraySize; i++)
    {
        u[i] = 0;
        u1[i] = 0;
        u2[i] = 0;
    }
 
    // set value of u1 2/2 to 1
    u1[arraySize / 2 + size / 2] = 1.0;
 
    // allocate cuda memory for u, u1 and u2 and copy the original ones in them
    float *d_u, *d_u1, *d_u2;
    cudaMalloc(&d_u, arrayMemSize);
    cudaMalloc(&d_u1, arrayMemSize);
    cudaMalloc(&d_u2, arrayMemSize);
    cudaMemcpy(d_u, u, arrayMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u1, u1, arrayMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2, u2, arrayMemSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < numIterations; i++)
    {
        finiteDrumUpdate<<<numBlocks, threadsPerBlock>>>(d_u, d_u1, d_u2, arraySize, size, numBlocks, threadsPerBlock);

        // for every iteration update the arrays
        float *temp = d_u2;
        d_u2 = d_u1;
        d_u1 = d_u;
        d_u = temp;
    }
 
    free(u);
    free(u1);
    free(u2);
    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);}

int main(int argc, char* argv[]) {
    
    if (argc != 2) {
		printf("Please include variables.\n");
		return 1;
	  }

	  int numIterations = atoi(argv[1]);
    clock_t start;
    printf("\nQ2:\n");
    start = clock();
    //one block, one thread per node
    process(4, numIterations, 1, 16);
    printf("\nTime = %d\n", clock() - start);

    return 0;
}