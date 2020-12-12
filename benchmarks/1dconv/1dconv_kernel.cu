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
            if(inputIndex >= 0 && inputIndex < arraySize) {
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
