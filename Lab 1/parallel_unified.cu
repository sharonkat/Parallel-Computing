// %%writefile parallel_unified.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

# define AND 0
# define OR 1
# define NAND 2
# define NOR 3
# define XOR 4
# define XNOR 5

__global__ void logic_gates(int num_threads, int num_blocks, char* old_list, char* new_list) {

    int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int i = global_thread_id;
    bool result;

    if (i < num_threads*num_blocks) {
        bool input_a = old_list[i*3];
        bool input_b = old_list[i*3+1];
        int gate = old_list[i*3+2];

        switch (gate) {
            case AND:
                result = (input_a && input_b);
                break;
            case OR:
                result = (input_a || input_b);
                break;
            case NAND:
                result = !(input_a && input_b);
                break;
            case NOR:
                result = !(input_a || input_b);
                break;
            case XOR:
                result = (input_a ^ input_b);
                break;
            case XNOR:
                result = !(input_a ^ input_b);
                break;
            default:
                printf("ERROR: Input gate invalid.\n");
                break;
        }
        new_list[i] = result;
    }
}

char* readFile(char *filename, int size_of_list, char* old_list);

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Error: program expects 3 agruments.\n");
        return 1;
    }

    // File management
    char *file_path = argv[1];
    int file_length = atoi(argv[2]);
    char *output_file_path = argv[3];
    FILE *fptr;

    // Definitions
    char *new_list, *old_list;
    // Sizing values
    int number_of_vals = file_length * 3;
    int number_of_lines = file_length;
    double num_threads, num_blocks;

    clock_t start_time = clock();

    // Allocation memory on GPU for old list and new list
    cudaMallocManaged((void**)&old_list, number_of_vals*sizeof(char*));
    cudaMallocManaged((void**)&new_list, number_of_lines*sizeof(char*));

    // Convert the csv file to an array of integers
    old_list = readFile(file_path, number_of_lines, old_list);

    // Find how many blocks we need, and how many threads per block
    if (file_length > 1024) {
        num_blocks = ceil((double)number_of_lines / 1024.0);
        num_threads = ceil((double)number_of_lines / num_blocks);
    } else {
        num_threads = number_of_lines;
        num_blocks = 1;
    }

    // Launch kernel
    logic_gates<<<num_threads, num_blocks>>>(num_threads, num_blocks, old_list, new_list);
    cudaDeviceSynchronize();

    // Open pointer to write to output file
    fptr = fopen(output_file_path,"w");
    if(fptr == NULL)
    {
      printf("Error!");   
      exit(1);             
    }

    for (int i=0; i<file_length; i++){
        fprintf(fptr,"%d\n",new_list[i]);
    }

    cudaFree(old_list);
    cudaFree(new_list);
    fclose(fptr);

    clock_t end_time = clock();
	long total_time = (end_time - start_time);
    printf("Time recorded: %ld.\n", total_time);

    return 0;
}

char* readFile(char *filename, int size_of_list, char* old_list)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Encountered error when trying to open file.\n");
        return NULL;
    }
    printf("Reading file...\n");
    const char s[2] = ",";
    char *token;

    // char* list_of_vals = (char*)malloc(size_of_list*3*sizeof(int));

    int array_counter=0;
    if(fp != NULL)
    {
        char line[20];
        while(fgets(line, sizeof line, fp) != NULL)
        {
            token = strtok(line, s);
            // Using token=',' let us read the file and put it into the array
            while( token != NULL ) {
                //printf("Array Counter: %d\n", array_counter);
                old_list[array_counter] = atoi(token);
                array_counter++;

                token = strtok(NULL, s);
            }
        }

        fclose(fp);
    } else {
        perror(filename);
    }
    return old_list;

}