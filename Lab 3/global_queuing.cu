// %%writefile global_queuing.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

        __device__ int
        gate_solver(int gate, int input_node, int input_neighbor)
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

__global__ void global_queuing(int numBlocks, int threadsPerBlock, int numCurrLevelNodes, int *currLevelNodes_h, int *nodePtrs_h, int *nodeNeighbors_h,
                               int *nodeVisited_h, int *nodeOutput_h, int *nodeGate_h, int *nodeInput_h, int *nextLevelNodes_h, int *numNextLevelNodes_h)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    

    // loop over each node to find neighbours
    for (int index = thread_id; index < numCurrLevelNodes; index += threadsPerBlock * numBlocks)
    {
        int currNode = currLevelNodes_h[index];

        // get all the neighbours of the currently checked node
        for (int j = nodePtrs_h[currNode]; j < nodePtrs_h[currNode + 1]; j++)
        {
            int nb = nodeNeighbors_h[j];
            // check if we've been here before
            if (nodeVisited_h[nb] == 0)
            {
                // if not, set it to 1 to signalize we've been here
                nodeVisited_h[nb] = 1;
                // solve using the gate solver
                nodeOutput_h[nb] = gate_solver(nodeGate_h[nb], nodeOutput_h[currNode], nodeInput_h[nb]);

                // add to the global queue
                int pos = atomicAdd(numNextLevelNodes_h, 1);
                atomicExch(&nextLevelNodes_h[pos], nb);
            }
        }
    }
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
    //printf("Read input\n");
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
    //printf("Read input 3\n");
    fclose(fp);
    return len;
}

int main(int argc, char *argv[])
{

    // printf("beginning of main\n");
    if (argc != 7)
    {
        printf("Error: program expects 6 agruments.\n");
        return 1;
    }
    // Input filepaths
    char *nodePtrs_filepath = argv[1];
    char *nodeNeighbors_filepath = argv[2];
    char *nodeLinks_filepath = argv[3];
    char *currLevelNodes_filepath = argv[4];

    // Output filepaths
    char *nodeOutput = argv[5];
    char *nextLevelNodesOutput = argv[6];

    // Variables
    int numNodePtrs, numNodes;
    int *nodePtrs_h, *nodeNeighbors_h;
    int *nodeGate_h, *nodeInput_h, *nodeOutput_h, *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;

    int address = 0;
    int *numNextLevelNodes_h = &address;
    
    //output
    int *nextLevelNodes_h;
    // NB: hack assignments
    nextLevelNodes_h = (int *)malloc(40101 * 2 * sizeof(int));

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

    global_queuing<<<25, 64>>>(25, 64, numCurrLevelNodes, d_currLevelNodes_h, d_nodePtrs_h, d_nodeNeighbors_h, d_nodeVisited_h, d_nodeOutput_h, d_nodeGate_h, d_nodeInput_h, d_nextLevelNodes_h, d_numNextLevelNodes_h);

    // get the results
    cudaMemcpy(numNextLevelNodes_h, d_numNextLevelNodes_h, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextLevelNodes_h, d_nextLevelNodes_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeOutput_h, d_nodeOutput_h, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // output information
    FILE *fptr;
    fptr = fopen(nodeOutput, "w");
    if (fptr == NULL)
    {
        printf("Error: unable to open node output file.");
        exit(1);
    }

    fprintf(fptr, "%d\n", numNodes);

    for (int j = 0; j < numNodes; j++)
    {
        fprintf(fptr, "%d\n", nodeOutput_h[j]);
    }

    FILE *fptr2;
    fptr2 = fopen(nextLevelNodesOutput, "w");
    if (fptr2 == NULL)
    {
        printf("Error: unable to open next level output file.");
        exit(1);
    }

    fprintf(fptr2, "%d\n", *numNextLevelNodes_h);

    for (int k = 0; k < (*numNextLevelNodes_h); k++)
    {
        fprintf(fptr2, "%d\n", nextLevelNodes_h[k]);
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