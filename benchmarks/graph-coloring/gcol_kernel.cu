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
 * Implementation of Parallel Graph Colouring
 *
 * Based on:
 * M. Deveci, E. G. Boman, K. D. Devine and S. Rajamanickam
 * Parallel Graph Coloring for Manycore Architectures
 * 2016 IEEE International Parallel and Distributed Processing Symposium (IPDPS)
 *
 * Modified to enable "Work-Stealing" between blocks
 * Assumption:
 *   1) Edges are inputted with lowest index vertex first
 *
 ********************************************************************************************/
 
 
/**************************************************
 *
 *               KERNEL FUNCTIONS
 *
 **************************************************/

__global__ void assignColorsKernel(int *head, int *tail, int *colorSet, int *vertexColor, 
    int *VForbidden, int *TVForbidden, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0) {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last) {
        if(tid + my_base < n_t_last) {    
            if(vertexColor[tid + my_base] < 0)
                vertexColor[tid + my_base] = -1 * vertexColor[tid + my_base];
            else if(vertexColor[tid + my_base] == 0) {
            
                int allForbid = VForbidden[tid + my_base] | TVForbidden[tid + my_base];
                
                // Can probably be optimized using bitwise operations
                bool flag = false;
                for(int i = 0; i < 32; ++i) {
                    if((allForbid & (1 << i)) == 0) {
                        vertexColor[tid + my_base] = i + 1;
                        flag = true;
                        break;
                    }
                }
                
                if(!flag) {
                    bool flag2 = false;
                    // Can probably be optimized using bitwise operations
                    for(int i = 0; i < 32; ++i) {
                        if((VForbidden[tid + my_base] & (1<<i)) == 0) {
                            flag2 = true;
                            break;
                        }
                    }
                    if(!flag2) {
                        colorSet[tid + my_base]++;
                        VForbidden[tid + my_base] = 0;
                        TVForbidden[tid + my_base] = 0;
                    }
                }
            }
        }
        __syncthreads();

        if(tid == 0) {
#ifdef RACEY
            *base = atomicAdd_block(&head[bid], NTHREADS);
#else
            *base = atomicAdd(&head[bid], NTHREADS);
#endif
        }
#ifdef RACEY
#else
        __syncthreads();
#endif
        my_base = *base;

        if(*base < n_t_last)
            continue;
        __syncthreads();

        if (tid == 0) {
            int otherBlock = 0;
            for (int block = (bid + 1);
                block < (bid + NBLOCKS); block++) {
                otherBlock = block % NBLOCKS;
                int h = atomicAdd(&head[otherBlock], 0);
                int t = tail[otherBlock];
                if ((h + NTHREADS) < t) {
                    break;
                }
            }
            *base = atomicAdd(&head[otherBlock], NTHREADS);
            atomicExch_block(blockId, otherBlock);
        }
        __syncthreads();
        bid = atomicAdd_block(blockId, 0);
        my_base = *base;
        n_t_last = tail[bid];
    }
}

__global__ void detectConflictsKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, 
    int *colorSet, int *vertexColor, int *complete, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0) {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last) {
        if(tid + my_base < n_t_last) {
            int U = edgeSetU[tid + my_base];
            int V = edgeSetV[tid + my_base];

            if(colorSet[U] == colorSet[V] && 
                atomicAdd(&vertexColor[U], 0) == atomicAdd(&vertexColor[V], 0)) {
                atomicExch(&vertexColor[U], 0);
            }

            if(atomicAdd(&vertexColor[U], 0) == 0 || atomicAdd(&vertexColor[V], 0) == 0) {
                atomicExch(complete, 0);
            }
        }
        __syncthreads();

        if(tid == 0) {
#ifdef RACEY
            *base = atomicAdd_block(&head[bid], NTHREADS);
#else
            *base = atomicAdd(&head[bid], NTHREADS);
#endif
        }
#ifdef RACEY
#else
        __syncthreads();
#endif
        my_base = *base;

        if(*base < n_t_last)
            continue;
        __syncthreads();

        if (tid == 0) {
            int otherBlock = 0;
            for (int block = (bid + 1);
                block < (bid + NBLOCKS); block++) {
                otherBlock = block % NBLOCKS;
                int h = atomicAdd(&head[otherBlock], 0);
                int t = tail[otherBlock];
                if ((h + NTHREADS) < t) {
                    break;
                }
            }
            *base = atomicAdd(&head[otherBlock], NTHREADS);
            atomicExch_block(blockId, otherBlock);
        }
        __syncthreads();
        bid = atomicAdd_block(blockId, 0);
        my_base = *base;
        n_t_last = tail[bid];
    }
}

__global__ void forbidColorsKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, 
    int *colorSet, int *vertexColor, int *VForbidden, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0) {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last) {
        if(tid + my_base < n_t_last) {    
            int U = edgeSetU[tid + my_base];
            int V = edgeSetV[tid + my_base];
            
            if(colorSet[U] == colorSet[V]) {
                if(vertexColor[U] > 0 && vertexColor[V] == 0) {
                    atomicOr(&VForbidden[V], (1 << (vertexColor[U] - 1)));
                }

                if(vertexColor[U] == 0 && vertexColor[V] > 0) {
                    atomicOr(&VForbidden[U], (1 << (vertexColor[V] - 1)));
                }
            }
    
        }
        __syncthreads();

        if(tid == 0) {
#ifdef RACEY
            *base = atomicAdd_block(&head[bid], NTHREADS);
#else
            *base = atomicAdd(&head[bid], NTHREADS);
#endif
        }
#ifdef RACEY
#else
        __syncthreads();
#endif
        my_base = *base;

        if(*base < n_t_last)
            continue;
        __syncthreads();

        if (tid == 0) {
            int otherBlock = 0;
            for (int block = (bid + 1);
                block < (bid + NBLOCKS); block++) {
                otherBlock = block % NBLOCKS;
                int h = atomicAdd(&head[otherBlock], 0);
                int t = tail[otherBlock];
                if ((h + NTHREADS) < t) {
                    break;
                }
            }
            *base = atomicAdd(&head[otherBlock], NTHREADS);
            atomicExch_block(blockId, otherBlock);
        }
        __syncthreads();
        bid = atomicAdd_block(blockId, 0);
        my_base = *base;
        n_t_last = tail[bid];
    }
}
