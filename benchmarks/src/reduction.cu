 /********************************************************************************************
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 ********************************************************************************************/
 /******************************************************************************************** 
 * Implementation of Reduction
 *
 * Edited by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 * Each block contains a subarray which it is responsible for reducing using block scope.
 * Original code has been modified to use atomics and threadfences instead of syncthreads.
 * 
 ********************************************************************************************/
/*
    Parallel reduction kernels
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

#define DATATYPE int

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}


__device__ unsigned int retirementCount = 0;
/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile DATATYPE *sdata, int *lock, DATATYPE mySum, const unsigned int tid)
{
    sdata[tid] = mySum;
    lock[tid] = 0;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256)
    {
        if (tid >= 128)
        {
            sdata[tid] = mySum + sdata[tid + 256];
            __threadfence_block();
            atomicExch_block(&lock[tid], 1);
        }
        else if(tid < 128)
        {
            mySum = mySum + sdata[tid + 256];
        }
    }

    if (blockSize >= 256 && tid < 128)
    {
        while(blockSize >= 512 && atomicAdd_block(&lock[tid + 128], 0) != 1);
        __threadfence_block();
        
        if(tid >= 64)
        {
            sdata[tid] = mySum + sdata[tid + 128];
            __threadfence_block();
            atomicExch_block(&lock[tid], 1);
        }
        else if(tid < 64)
        {
            mySum = mySum + sdata[tid + 128];
        }
        
    }

    if (blockSize >= 128 && tid < 64)
    {
        while(blockSize >= 256 && atomicAdd_block(&lock[tid + 64], 0) != 1);
        __threadfence_block();
        if (tid >= 32)
        {
            sdata[tid] = mySum + sdata[tid +  64];
#ifdef RACEY
#else
            __threadfence_block();
#endif
            atomicExch_block(&lock[tid], 1);
        }
        else if(tid < 32)
        {
            mySum = mySum + sdata[tid + 64];
        }
    }

    if (tid < 32)
    {
        while(blockSize >= 128 && atomicAdd_block(&lock[tid + 32], 0) != 1);
        __threadfence_block();
        
        if (blockSize >=  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        if (blockSize >=  32 && tid < 16)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        if (blockSize >=  16 && tid < 8)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        if (blockSize >=   8 && tid < 4)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        if (blockSize >=   4 && tid < 2)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        if (blockSize >=   2 && tid == 0)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  1];
        }
    }
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void
reduceBlocks(const DATATYPE *g_idata, volatile DATATYPE *g_odata, volatile DATATYPE *sdata, int *locks, unsigned int n)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    DATATYPE mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }
    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, locks, mySum, tid);
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const DATATYPE *g_idata, volatile DATATYPE *g_odata, volatile DATATYPE *sdata, int *locks, unsigned int n)
{
    __shared__ bool amLast;
    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, &sdata[blockIdx.x * blockSize], &locks[blockIdx.x * blockSize], n);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;

        // wait until all outstanding memory instructions in this thread are finished

#ifdef RACEY
        __threadfence_block();
#else
        __threadfence();
#endif
        // Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        __syncthreads();

        // The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            DATATYPE mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(&sdata[blockIdx.x * blockSize], &locks[blockIdx.x * blockSize], mySum, tid);

            if (tid==0)
            {
                g_odata[0] = sdata[blockIdx.x * blockSize];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

void reduceSinglePasses(int size, DATATYPE *d_idata, DATATYPE *d_odata, int nThreads, int nBlocks, DATATYPE *smem, int *locks)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
      
    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (nThreads)
        {
            case 512:
                reduceSinglePass<512, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 256:
                reduceSinglePass<256, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 128:
                reduceSinglePass<128, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 64:
                reduceSinglePass< 64, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 32:
                reduceSinglePass< 32, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 16:
                reduceSinglePass< 16, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  8:
                reduceSinglePass<  8, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  4:
                reduceSinglePass<  4, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  2:
                reduceSinglePass<  2, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  1:
                reduceSinglePass<  1, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;
        }
    }
    else
    {
        switch (nThreads)
        {
            case 512:
                reduceSinglePass<512, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 256:
                reduceSinglePass<256, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 128:
                reduceSinglePass<128, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 64:
                reduceSinglePass< 64, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 32:
                reduceSinglePass< 32, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case 16:
                reduceSinglePass< 16, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  8:
                reduceSinglePass<  8, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  4:
                reduceSinglePass<  4, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  2:
                reduceSinglePass<  2, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;

            case  1:
                reduceSinglePass<  1, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, locks, size);
                break;
        }
    }
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1)
    {
        threads = 1;
        blocks = 1;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
        blocks = max(1, n / (threads * 2));
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int size;    // number of elements to reduce
    cin >> size;
    
    DATATYPE *h_idata = (DATATYPE *)malloc(size * sizeof(DATATYPE));

    for(int i = 0; i < size; ++i)
        cin >> h_idata[i];

    int nBlocks, nThreads;
    getNumBlocksAndThreads(size, NBLOCKS, NTHREADS, nBlocks, nThreads);

    // allocate mem for the result on host side
    DATATYPE *h_odata = (DATATYPE *) malloc(nBlocks*sizeof(DATATYPE));

    // allocate device memory and data
    int *d_idata = NULL;
    int *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **) &d_idata, size * sizeof(DATATYPE)));
    checkCudaErrors(cudaMalloc((void **) &d_odata, nBlocks*sizeof(DATATYPE)));
    
    int smemSize = nThreads * nBlocks;
    DATATYPE *d_smem;
    int *d_locks;
    checkCudaErrors(cudaMalloc((void **) &d_smem, smemSize * sizeof(DATATYPE)));
    checkCudaErrors(cudaMalloc((void **) &d_locks, smemSize * sizeof(int)));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, size * sizeof(DATATYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, nBlocks * sizeof(DATATYPE), cudaMemcpyHostToDevice));

    // execute the kernel
    reduceSinglePasses(size, d_idata, d_odata, nThreads, nBlocks, d_smem, d_locks);

    // copy final sum from device to host
    DATATYPE gpu_result = 0;
    
    checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(DATATYPE), cudaMemcpyDeviceToHost));

    cout << "GPU result = " << gpu_result << "\n";

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    return 0;
}
