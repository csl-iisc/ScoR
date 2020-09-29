/********************************************************************************************
 * Copyright (c) 2020 Indian Institute of Science
 * All rights reserved.
 *
 * Developed by:    Alvin George A.
 *                  Computer Systems Lab
 *                  Indian Institute of Science
 *                  https://csl.csa.iisc.ac.in/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of Computer Systems Lab, Indian Institute of Science, 
 *        nor the names of its contributors may be used to endorse or promote products 
 *        derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;
#include "1dconv_kernel.cuh"
 
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
