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
 * Implementation of Matrix Multiplication using Threadfence Locks
 *
 * Each block contains a subpart of a row of the input matrix.
 *
 ********************************************************************************************/

#include "mm_kernel.cuh"

/***************************************************
 *
 *               KERNEL FUNCTIONS
 *
 ***************************************************/

/***************************************************
 *                  INPUT
 * 
 * Matrix A, Matrix B, Result Matrix C, 
 * # rows of A (= # rows of C), 
 * # columns of A (= # rows of B), 
 * # columns of B (= # columns of C)
 *
 * All in row-major format
 ***************************************************/
__global__ void matMultKernel(datatype *A, datatype *B, volatile datatype *C, 
    datatype rA, datatype cA, datatype cB, int *sh_locks, volatile datatype *tempCs, volatile int *gl_lock)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Allow each warp its own lock for better parallelism
    int &sh_lock = sh_locks[bid * WARP_SIZE + (tid % WARP_SIZE)];
    volatile datatype &tempC = tempCs[bid * WARP_SIZE + (tid % WARP_SIZE)];

    int offset = 0;

    if(tid == 0)
        sh_lock = 0;
    __syncthreads();
    
    // Repeat until entire matrix has been covered
    while(offset / ((cA + blockDim.x - 1) / blockDim.x) < rA) {
        int row = (blockIdx.x + offset) / ((cA + blockDim.x - 1) / blockDim.x);
        // Iterate over columns in matrix B
        for(int i = 0; i < (cB + WARP_SIZE - 1) / WARP_SIZE; ++i) {
            // Initialize shared variables
            if(tid / WARP_SIZE == 0) {
                tempC = 0;
            }
            __syncthreads();
            for(int j = 0; j < WARP_SIZE; ++j) {
                int col = ((blockIdx.x + offset) % ((cA + blockDim.x - 1) / blockDim.x)) 
                          * blockDim.x + (tid / WARP_SIZE) * WARP_SIZE + j;
                datatype t = 0;
                if(row < rA && col < cA && (i *  WARP_SIZE) + (tid % WARP_SIZE) < cB) {
                    // Perform multiplication
                    t = A[col + row * cA] * B[(i *  WARP_SIZE) + (tid % WARP_SIZE) + col * cB];
                }
                
                // Have each thread update shared variable with value
                bool success = false;
                do {
                    if(atomicCAS_block(&sh_lock, 0, 1) == 0) {
#ifdef RACEY
#else
                        __threadfence_block();
#endif
                        tempC += t;
#ifdef RACEY
#else
                        __threadfence_block();
#endif
                        atomicExch_block(&sh_lock, 0);
                        success = true;
                    }
                }while(!success);
            }
            __syncthreads();
            
            // Update the result array with value
            if(tid / WARP_SIZE == 0 && row < rA && i * WARP_SIZE + tid < cB) {
                bool successful = false;
                do {
#ifdef RACEY
                    // 2 races: One on gl_lock due to block atomic, other on C, due to block-scope locking
                    if(0 == atomicCAS_block((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % NTHREADS], 0, 1)) {
                        __threadfence_block();
#else
                    if(0 == atomicCAS((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % NTHREADS], 0, 1)) {
                        __threadfence();
#endif
                        C[i * WARP_SIZE + tid + row * cB] += tempC;__threadfence();
                        atomicExch((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % NTHREADS], 0);
                        successful = true;
                    }
                }while(!successful);
            }
        }
        
        offset += gridDim.x;
    }
}
