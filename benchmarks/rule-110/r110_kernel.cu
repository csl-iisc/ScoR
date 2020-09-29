/********************************************************************************************
 * Copyright (c) 2020 Indian Institute of Science
 * All rights reserved.
 *
 * Developed by:    Aditya K Kamath
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

 /********************************************************************************************
 * Implementation of Rule 110 Cellular Automaton
 * https://en.wikipedia.org/wiki/Rule_110
 ********************************************************************************************/

#include "r110_kernel.cuh"

__device__ volatile int gl_lock = 0;

/***************************************************
 *
 *               DEVICE FUNCTIONS
 *
 ***************************************************/
__device__ bool isBorder()
{
    return threadIdx.x == 0 || threadIdx.x == blockDim.x - 1;
}

/*
 * From Wikipedia:
 * Current pattern           111 110 101 100 011 010 001 000
 * New state for center cell  0   1   1   0   1   1   1   0
 */
/***************************************************
 *
 *               KERNEL FUNCTIONS
 *
 ***************************************************/

__global__ void rule110Kernel(DATA *arr, volatile DATA *copy, int *comp, const int size, int step)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int btid = threadIdx.x;
    int offset = 0;
    while(offset < size) {
        int ind = tid + offset;
        if(ind < size) {
            // Copy array value into a copy
            copy[ind] = arr[ind];
        }
        offset += NBLOCKS * NTHREADS;
    }
    // Threadfence and mark complete
    if(isBorder()) {
#ifdef RACEY
        __threadfence_block();
#else
        __threadfence();
#endif
        atomicExch(&comp[tid], step);
    }
    else {
        __threadfence_block();
        atomicExch_block(&comp[tid], step);
    }
    
    // Use device scope for border neighbors, and block for non-border
    if(btid - 1 == 0 || btid == 0) {
        while(step != atomicAdd(&comp[(NBLOCKS * NTHREADS + tid - 1) % (NBLOCKS * NTHREADS)], 0));
        __threadfence();
    }
    else {
        while(step != atomicAdd_block(&comp[tid - 1], 0));
        __threadfence_block();
    }
    
    offset = 0;
    while(offset < size) {
        int ind = tid + offset;
        
        if(ind < size) {
            if(ind > 0)
                arr[ind] = copy[ind - 1];
            else
                arr[ind] = 0;
        }
        offset += NBLOCKS * NTHREADS;
    }
    
    if(btid + 1 == NTHREADS || btid + 1 == NTHREADS - 1) {
#ifdef RACEY
        while(step != atomicAdd_block(&comp[(NBLOCKS * NTHREADS + tid + 1) % (NBLOCKS * NTHREADS)], 0));
#else
        while(step != atomicAdd(&comp[(NBLOCKS * NTHREADS + tid + 1) % (NBLOCKS * NTHREADS)], 0));
#endif
        __threadfence();
    }
    else {
        while(step != atomicAdd_block(&comp[tid + 1], 0));
        __threadfence_block();
    }
    
    // Calculate new value of array and update
    offset = 0;
    while(offset < size) {
        int ind = tid + offset;
        if(ind < size) {
            int val = arr[ind];
            val <<= 1;
            if(ind < size)
                val += copy[ind];
            val <<= 1;

            if(ind + 1 < size)
                val += copy[ind + 1];
            // Update value based on neighbors
            switch(val) {
                case 7:
                case 4:
                case 0:
                    val = 0;
                break;
                case 6:
                case 5:
                case 3:
                case 2:
                case 1:
                    val = 1;
                break;                        
            }
            // Calculate new value and store
            arr[ind] = val;
        }
        offset += NBLOCKS * NTHREADS;
    }
}
