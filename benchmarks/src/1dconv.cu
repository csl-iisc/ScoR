/********************************************************************************************
 * Author:
 * Alvin George A., Indian Institute of Science
 ********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;

#define WARP_SIZE (NTHREADS < 32 ? NTHREADS : 32)

#define NTHREADS_TOT  (NBLOCKS * NTHREADS)

__global__
void initKernel(float *filter, int filterSize, float *array, int arraySize, float *output)
{
    int bid  = blockIdx.x;
    int btid = threadIdx.x;
    int gtid = bid * blockDim.x + btid;
    
    int yid = gtid;
    while (yid < arraySize) {
        output[yid] = 0.0;
        yid += NTHREADS_TOT;
    }
}

__global__
void convolveKernel(float *filter, int filterSize, float *array, int arraySize, float *output)
{
    int bid = blockIdx.x;
    int btid = threadIdx.x;
    int gtid = bid * blockDim.x + btid;
    
    int offset = 0;
    while(offset < arraySize * filterSize) {
        int filterIndex = (offset + gtid) % filterSize;
        int outputIndex = (offset + gtid) / filterSize; 
        if(outputIndex < arraySize) {
            int inputIndex = outputIndex - (filterIndex - filterSize / 2);
            if(inputIndex < arraySize) {
#ifdef RACEY
                atomicAdd_block(&output[outputIndex], array[inputIndex] * filter[filterIndex]);
#else
                // Only this block's threads are writing to location
                if(btid >= filterIndex && blockDim.x - btid >= filterSize - filterIndex)
                    atomicAdd_block(&output[outputIndex], array[inputIndex] * filter[filterIndex]);
                else
                    atomicAdd(&output[outputIndex], array[inputIndex] * filter[filterIndex]);
#endif
            }
        }
        offset += NTHREADS_TOT;
    }
}


float *filter;
int filterSize = 0;

float *array;
int arraySize = 0;

float *output;

void print_arr(float *x, int arraySize, const char *s)
{
    printf("%s:\n", s);
    for (int i=0; i < arraySize; i++)
        printf("[%d]: %0.2f\n", i, x[i]);
    printf("\n-------\n");
}

void read_arr(float *x, int len)
{
    for(int i = 0; i < len; ++i)
        cin >> x[i];
}

void init_inputs_outputs()
{
    cin >> filterSize >> arraySize;

    filter = (float *) malloc(filterSize * sizeof(float));
    array = (float *) malloc(arraySize * sizeof(float));
    output = (float *) malloc(arraySize * sizeof(float));
    if (filter == NULL || array == NULL || output == NULL)
        exit(1);

    read_arr(filter, filterSize);
    read_arr(array, arraySize);
}

void write_output(float *x, int len)
{
    FILE *ofp = fopen("output.txt", "w");
    fprintf(ofp, "%d, ", len);
    for (int i=0; i < len; i++)
        fprintf(ofp, "%0.2f, ", x[i]);
    fclose(ofp);
}


void cudaerrCheck(cudaError_t err, int line, const char *msg)
{
    if (err != cudaSuccess) {
        printf("CudaError line %d: %s\n", line, msg);
        exit(1);
    }
}

#define CUDAERRCHECK(e, m)      cudaerrCheck(e, __LINE__, m)


int main()
{
    float *d_filter, *d_array, *d_output;
    init_inputs_outputs();

    CUDAERRCHECK(cudaMalloc(&d_filter, sizeof(float) * filterSize), "Malloc device h");
    CUDAERRCHECK(cudaMalloc(&d_array, sizeof(float) * arraySize), "Malloc device x");
    CUDAERRCHECK(cudaMalloc(&d_output, sizeof(float) * arraySize), "Malloc device y");
    CUDAERRCHECK(cudaMemcpy(d_filter, filter, sizeof(float) * filterSize, cudaMemcpyHostToDevice), "Memcpy device h");
    CUDAERRCHECK(cudaMemcpy(d_array, array, sizeof(float) * arraySize, cudaMemcpyHostToDevice), "Memcpy device x");

    initKernel <<<NBLOCKS, NTHREADS>>> (d_filter, filterSize, d_array, arraySize, d_output);    
    convolveKernel <<<NBLOCKS, NTHREADS>>> (d_filter, filterSize, d_array, arraySize, d_output);
    
    CUDAERRCHECK(cudaMemcpy(output, d_output, sizeof(float) * arraySize, cudaMemcpyDeviceToHost), "Memcpy host y");
#if 0
    print_arr(output, arraySize, "y");
#endif
#if 1
    write_output(output, arraySize);
#endif
}

