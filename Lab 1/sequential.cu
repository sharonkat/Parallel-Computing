// %%writefile sequential.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "gputimer.h"

# define AND 0
# define OR 1
# define NAND 2
# define NOR 3
# define XOR 4
# define XNOR 5

bool sequential(bool input_a, bool input_b, int input_gate) {
    bool result;

    switch (input_gate) {
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
            result = -1;
    }

    return result;
}

char* readFile(char *filename, int size_of_list);

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Error: program expects 3 agruments.\n");
        return 1;
    }

    char *file_path = argv[1];
    int file_length = atoi(argv[2]);
    char *output_file_path = argv[3];
    FILE *fptr;

    clock_t start_time = clock();

    // Malloc array for results
    char *result = (char*)malloc(file_length * sizeof(char*)); 


    // convert the csv file to an array of integers
    char* list_of_lines = readFile(file_path, file_length);

    fptr = fopen(output_file_path,"w");
    if(fptr == NULL)
    {
      printf("Error!");   
      exit(1);             
    }   

    // Now we need to send every 3 elements to sequential()
    for (int i=0; i<file_length*3; i+=3) {
        //printf("%d, %d, %d\n", list_of_lines[i], list_of_lines[i+1], list_of_lines[i+2]);
        result[i/3] = sequential((bool)list_of_lines[i], (bool)list_of_lines[i+1], list_of_lines[i+2]);
        // Write results in output file
        fprintf(fptr,"%d\n",result[i/3]);
    }
   
    free(result);
    fclose(fptr);

	clock_t end_time = clock();
	long total_time = (end_time - start_time);
    printf("Time recorded: %ld.\n", total_time);

    return 0;
}

char* readFile(char *filename, int size_of_list)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Encountered error when trying to open file.\n");
        return NULL;
    }
    printf("reading file...\n");
    const char s[2] = ",";
    char *token;

    char* list_of_lines = (char*)malloc(size_of_list*3*sizeof(int));

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
                list_of_lines[array_counter] = atoi(token);
                array_counter++;

                token = strtok(NULL, s);
            }
        }

        fclose(fp);
    } else {
        perror(filename);
    }
    return list_of_lines;

}