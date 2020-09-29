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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

#include "red_kernel.cu"

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "\n";
        exit(1);
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
