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
 * Implementation of Parallel Graph Connectivity Computation
 *
 * Based on:
 * M. Sutton, T. Ben-Nun and A. Barak
 * Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)
 *
 * Modified to enable "Work-Stealing" between blocks
 *
 ********************************************************************************************/

/**************************************************
 *
 *               KERNEL FUNCTIONS
 *
 **************************************************/

__global__ void initKernel(int *head, int *tail, int *graphComponents, int *bases, int *blockIds)
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
            graphComponents[tid + my_base] = tid + my_base;
        }
#ifdef RACEY
#else
        __syncthreads();
#endif

        if(tid == 0) {
            *base = atomicAdd(&head[bid], NTHREADS);
        }

        __syncthreads();
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

__global__ void linkKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, int *graphComponents, int *bases, int *blockIds)
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
            int p1 = atomicAdd(&graphComponents[U], 0);
            int p2 = atomicAdd(&graphComponents[V], 0);

            while (p1 != p2) {
                int maxi = p1 > p2 ? p1 : p2;
                int mini = p1 + (p2 - maxi);

                int prev = atomicCAS(&graphComponents[maxi], maxi, mini);

                if (prev == maxi || prev == mini) 
                    break;

                p1 = atomicAdd(&graphComponents[atomicAdd(&graphComponents[maxi], 0)], 0);
                p2 = atomicAdd(&graphComponents[mini], 0);
            }
        }
#ifdef RACEY
#else
        __syncthreads();
#endif


        if(tid == 0) {
            *base = atomicAdd(&head[bid], NTHREADS);
        }

        __syncthreads();
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
#ifdef RACEY
            *base = atomicAdd_block(&head[otherBlock], NTHREADS);
#else
            *base = atomicAdd(&head[otherBlock], NTHREADS);
#endif
            atomicExch_block(blockId, otherBlock);
        }
        __syncthreads();
        bid = atomicAdd_block(blockId, 0);
        my_base = *base;
        n_t_last = tail[bid];
    }
}

__global__ void compressKernel(int *head, int *tail, int *graphComponents, int *bases, int *blockIds)
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
            int current = atomicAdd(&graphComponents[tid + my_base], 0);
            int parent = atomicAdd(&graphComponents[current], 0);

            while (current != parent) {
                int val = atomicCAS(&graphComponents[tid + my_base], current, parent);
                if(val == current) {
                    // Successful swap, continue compression
                    current = parent;
                    parent = atomicAdd(&graphComponents[current], 0);
                }
                else {
                    // Unsuccessful swap, restart compression
                    current = atomicAdd(&graphComponents[tid + my_base], 0);
                    parent = atomicAdd(&graphComponents[current], 0);
                }
            }
 
        }
#ifdef RACEY
#else
        __syncthreads();
#endif
        if(tid == 0) {
            *base = atomicAdd(&head[bid], NTHREADS);
        }

        __syncthreads();
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
#ifdef RACEY
            *base = atomicAdd_block(&head[otherBlock], NTHREADS);
#else
            *base = atomicAdd(&head[otherBlock], NTHREADS);
#endif
            atomicExch_block(blockId, otherBlock);
        }
        __syncthreads();
        bid = atomicAdd_block(blockId, 0);
        my_base = *base;
        n_t_last = tail[bid];
    }
}
