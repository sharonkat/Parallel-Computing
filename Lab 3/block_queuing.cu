// %%writefile block_queuing.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5
// Block Variables
__shared__ int block_num;
__shared__ int actual_num;
__device__ int global_num;

__device__ int gate_solver(int gate, int input_node, int input_neighbor)
{
    bool result;

    switch (gate)
    {
    case AND:
        result = (input_node && input_neighbor);
        break;
    case OR:
        result = (input_node || input_neighbor);
        break;
    case NAND:
        result = !(input_node && input_neighbor);
        break;
    case NOR:
        result = !(input_node || input_neighbor);
        break;
    case XOR:
        result = (input_node ^ input_neighbor);
        break;
    case XNOR:
        result = !(input_node ^ input_neighbor);
        break;
    default:
        printf("ERROR: Input gate invalid.\n");
        return -1;
    }

    return (int)result;
}

__global__ void block_queuing(int iteration, int num_iterations, int numBlocks, int blockSize, int blockQueueCapacity, int numCurrLevelNodes, int *currLevelNodes_h,
                              int *nodePtrs_h, int *nodeNeighbors_h,int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h,
                              int *numNextLevelNodes_h)
{
    int queueSize = blockQueueCapacity;
    extern __shared__ int shared_memory_queue[];
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 0;

    for (int i = thread_id; i < numCurrLevelNodes; i += blockSize * numBlocks) {
        offset = iteration * blockSize * numBlocks + i;
        int node = currLevelNodes_h[offset];

        for (int j = nodePtrs_h[node]; j < nodePtrs_h[node + 1]; j++)
        {
            int neighbor = nodeNeighbors_h[j];

            if (nodeVisited_h[neighbor] == 0)
            {
                nodeVisited_h[neighbor] = 1;
                nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);

                int new_index = atomicAdd(&block_num, 1);
                if (new_index < queueSize) {
                    int actual_index = atomicAdd(&actual_num, 1);
                    shared_memory_queue[new_index] = neighbor;
                } else {
                    int global_index = atomicAdd(&global_num, 1);
                    nextLevelNodes_h[global_index] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    if ((threadIdx.x == 0) && (iteration == num_iterations - 1))
    {
        for (int j=0; j<actual_num; j++) {
            int glob = atomicAdd(&global_num, 1);
            nextLevelNodes_h[glob] = shared_memory_queue[j];
        }
    }
    numNextLevelNodes_h[0] = global_num;
}

void global_queuing(int numBlocks, int blockSize, int sharedQueueSize, int numCurrLevelNodes, int *currLevelNodes_h, int *nodePtrs_h,
                    int *nodeNeighbors_h, int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h, int *numNextLevelNodes_h)
{
    int iteration=0;
    int num_iterations = numCurrLevelNodes / (numBlocks * blockSize) +1;

    while (iteration < num_iterations) {
        block_queuing<<<numBlocks, blockSize>>>(iteration, num_iterations, numBlocks, blockSize, sharedQueueSize, numCurrLevelNodes, currLevelNodes_h, nodePtrs_h, nodeNeighbors_h, nodeVisited_h, nodeOutput_h, nodeGate_h, nodeInput_h, nextLevelNodes_h, numNextLevelNodes_h);
        cudaDeviceSynchronize();
        iteration++;
    }
    cudaDeviceSynchronize();
}

int read_input_one_two_four(int **input1, char *filepath)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = (int *)malloc(len * sizeof(int));

    int temp1;

    while (fscanf(fp, "%d", &temp1) == 1)
    {
        (*input1)[counter] = temp1;

        counter++;
    }

    fclose(fp);
    return len;
}

int read_input_three(int **input1, int **input2, int **input3, int **input4, char *filepath)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = (int *)malloc(len * sizeof(int));
    *input2 = (int *)malloc(len * sizeof(int));
    *input3 = (int *)malloc(len * sizeof(int));
    *input4 = (int *)malloc(len * sizeof(int));

    int temp1;
    int temp2;
    int temp3;
    int temp4;
    while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4)
    {
        (*input1)[counter] = temp1;
        (*input2)[counter] = temp2;
        (*input3)[counter] = temp3;
        (*input4)[counter] = temp4;
        counter++;
    }
    fclose(fp);
    return len;
}

int main(int argc, char *argv[])
{

    if (argc != 10)
    {
        printf("Error: program expects 9 agruments.\n");
        return 1;
    }
    // Block params
    int numBlocks = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    int sharedQueueSize = atoi(argv[3]);
    // Input filepaths
    char *nodePtrs_filepath = argv[4];
    char *nodeNeighbors_filepath = argv[5];
    char *nodeLinks_filepath = argv[6];
    char *currLevelNodes_filepath = argv[7];

    // Output filepaths
    char *nodeOutput = argv[8];
    char *nextLevelNodesOutput = argv[9];

    // Variables
    int numNodePtrs, numNodes;
    int *nodePtrs_h, *nodeNeighbors_h;
    int *nodeGate_h, *nodeInput_h, *nodeOutput_h, *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;

    // Output
    int *nextLevelNodes_h;
    nextLevelNodes_h = (int *)malloc(40101 * 2 * sizeof(int));
    int numNextLevelNodes_h = 0;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, nodePtrs_filepath);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, nodeNeighbors_filepath);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, nodeLinks_filepath);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, currLevelNodes_filepath);

    int *d_numNextLevelNodes_h, *d_nodePtrs_h, *d_nodeNeighbors_h, *d_nodeVisited_h, *d_nodeGate_h, *d_nodeInput_h, *d_nodeOutput_h, *d_currLevelNodes_h, *d_nextLevelNodes_h;

    // CUDA memory allocation
    cudaMalloc((void **)&d_numNextLevelNodes_h, sizeof(int));
    cudaMalloc((void **)&d_nodePtrs_h, numNodePtrs * sizeof(int));
    cudaMalloc((void **)&d_nodeNeighbors_h, numTotalNeighbors_h * sizeof(int));
    cudaMalloc((void **)&d_nodeVisited_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeGate_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeInput_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_nodeOutput_h, numNodes * sizeof(int));
    cudaMalloc((void **)&d_currLevelNodes_h, numCurrLevelNodes * sizeof(int));
    cudaMalloc((void **)&d_nextLevelNodes_h, numTotalNeighbors_h * sizeof(int));

    // copy information
    cudaMemcpy(d_nodePtrs_h, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeNeighbors_h, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeVisited_h, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeGate_h, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeInput_h, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeOutput_h, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_currLevelNodes_h, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

    global_queuing(numBlocks, blockSize, sharedQueueSize, numCurrLevelNodes, d_currLevelNodes_h, d_nodePtrs_h, d_nodeNeighbors_h, d_nodeVisited_h, d_nodeOutput_h, d_nodeGate_h, d_nodeInput_h, d_nextLevelNodes_h, d_numNextLevelNodes_h);

    // get the results
    cudaMemcpy(&numNextLevelNodes_h, d_numNextLevelNodes_h, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextLevelNodes_h, d_nextLevelNodes_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeOutput_h, d_nodeOutput_h, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // output information
    FILE *fptr = fopen(nodeOutput, "w");
    if (fptr == NULL)
    {
        printf("Error: unable to open node output file.");
        exit(1);
    }

    for (int i = 0; i < numNodes + 1; i++)
    {
        if (i == 0)
        {
            fprintf(fptr, "%d\n", numNodes);
        }
        fprintf(fptr, "%d\n", nodeOutput_h[i]);
    }
    fclose(fptr);

    FILE *fptr2 = fopen(nextLevelNodesOutput, "w");
    if (fptr2 == NULL)
    {
        printf("Error: unable to open next level output file.");
        exit(1);
    }

    for (int i = 0; i < numNextLevelNodes_h + 1; i++)
    {
        if (i == 0)
        {
            fprintf(fptr, "%d\n", numNextLevelNodes_h);
        }
        fprintf(fptr2, "%d\n", nextLevelNodes_h[i]);
    }
     
    fclose(fptr);   
    fclose(fptr2);

    free(nodePtrs_h);
    free(nodeNeighbors_h);
    free(nodeGate_h);
    free(nodeInput_h);
    free(nodeOutput_h);
    free(nodeVisited_h);
    free(currLevelNodes_h);
    free(nextLevelNodes_h);

    cudaFree(d_nextLevelNodes_h);
    cudaFree(d_nodePtrs_h);
    cudaFree(d_nodeNeighbors_h);
    cudaFree(d_nodeVisited_h);
    cudaFree(d_currLevelNodes_h);
    cudaFree(d_nodeGate_h);
    cudaFree(d_nodeInput_h);
    cudaFree(d_nodeOutput_h);

    return 0;
}