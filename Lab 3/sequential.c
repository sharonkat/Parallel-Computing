//%%writefile sequential.c

#include <stdio.h>
#include <stdlib.h>
# define AND 0
# define OR 1
# define NAND 2
# define NOR 3
# define XOR 4
# define XNOR 5

int gate_solver(int gate, int input_node, int input_neighbor) {
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

    return (int) result;
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

    //output
    int *nextLevelNodes_h;
    // NB: hack assignments
    nextLevelNodes_h = (int *)malloc(40101 * 2 * sizeof(int));
    int numNextLevelNodes_h = 0;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, nodePtrs_filepath);

    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, nodeNeighbors_filepath);

    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, nodeLinks_filepath);

    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, currLevelNodes_filepath);

    // nextLevelNodes_h = (int *)malloc(numTotalNeighbors_h * numNodePtrs * numNodes * sizeof(int));

    int node, neighbor;

    for (int index = 0; index < numCurrLevelNodes; index++)
    {
        node = currLevelNodes_h[index];
        for (int number_index = nodePtrs_h[node]; number_index < nodePtrs_h[node + 1]; number_index++)
        {
            neighbor = nodeNeighbors_h[number_index];
            // If neighbor hasn't been visited yet
            if (nodeVisited_h[neighbor] == 0)
            {
                // Mark it and add it to the queue
                nodeVisited_h[neighbor] = 1;
                nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
                // NB: removed dereference for numNextLevelNodes
                nextLevelNodes_h[numNextLevelNodes_h] = neighbor;
                ++(numNextLevelNodes_h);
            }
        }
    }

    // Create output files
    FILE *fptr = fopen(nodeOutput, "w");
    if (fptr == NULL)
    {
        printf("Error: unable to open node output file.");
        exit(1);
    }

    for (int i = 0; i < numNodes + 1; i++)
    {
        if (i==0) {
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
        if (i==0) {
            fprintf(fptr, "%d\n", numNextLevelNodes_h);
        }
        fprintf(fptr2, "%d\n", nextLevelNodes_h[i]);
    }
    fclose(fptr2);

    free(nodePtrs_h);
    free(nodeNeighbors_h);
    free(nodeGate_h);
    free(nodeInput_h);
    free(nodeOutput_h);
    free(nodeVisited_h);
    free(currLevelNodes_h);
    free(nextLevelNodes_h);
    return 0;
}