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
 * Implementation of Unbalanced Tree Search
 *
 * Based on:
 * S. Olivier, J. Huan, J. Liu, J. Prins, J. Dinan, P. Sadayappan, C. Tseng
 * UTS: An Unbalanced Tree Search Benchmarks
 * 2006 International Workshop on Languages and Compilers for Parallel Computing (LCPC)
 * 
 ********************************************************************************************/

#include <stdio.h>
#include "uts_kernel.cuh"

int main(int argc, char *argv[])
{
    //parseParams(argc, argv);
    StackStats *d_ss1, *d_ss2, *h_ss1, *h_ss2;
    Node *localStacks, *stealStacks;
    
    Config config;
    scanf("%d %d", &config.maxHeight, &config.avgChildren);
    
    cudaMalloc((void **)&d_ss1, sizeof(StackStats) * NBLOCKS);
    cudaMalloc((void **)&d_ss2, sizeof(StackStats) * NBLOCKS);
    cudaMalloc((void **)&localStacks, sizeof(Node) * (NBLOCKS * LOCAL_DEPTH));
    cudaMalloc((void **)&stealStacks, sizeof(Node) * (NBLOCKS * MAXSTACKDEPTH));
    
    h_ss1 = (StackStats *)malloc(sizeof(StackStats) * NBLOCKS);
    for(int i = 0; i < NBLOCKS; ++i) {
        h_ss1[i].stackSize = LOCAL_DEPTH;
        h_ss1[i].workAvail = 1;
        h_ss1[i].top = LOCAL_DEPTH * i;
        h_ss1[i].totalNodes = 1;
        h_ss1[i].totalLeaves = 0;
        h_ss1[i].locked = 0;
    }
    
    h_ss2 = (StackStats *)malloc(sizeof(StackStats) * NBLOCKS);
    for(int i = 0; i < NBLOCKS; ++i) {
        h_ss2[i].stackSize = MAXSTACKDEPTH;
        h_ss2[i].workAvail = 0;
        h_ss2[i].top = MAXSTACKDEPTH * i;
        h_ss2[i].totalNodes = 0;
        h_ss2[i].totalLeaves = 0;
        h_ss2[i].locked = 0;
    }
    
    cudaMemcpy(d_ss1, h_ss1, sizeof(StackStats) * (NBLOCKS), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ss2, h_ss2, sizeof(StackStats) * (NBLOCKS), cudaMemcpyHostToDevice);
    
    Node root;
    root.param[HEIGHT] = 0;
    root.param[CHILDREN] = MAX_CHAR;
    int seed;
    scanf("%d", &seed);
    root.param[SEED] = (seed % MAX_CHAR);
    for(int i = 0; i < NBLOCKS; ++i) {
        root.param[NUMBER] = (i + 1) % MAX_CHAR;
        cudaMemcpy(&localStacks[LOCAL_DEPTH * i], &root, sizeof(Node), cudaMemcpyHostToDevice);
    }
    parTreeSearch<<<NBLOCKS, NTHREADS>>>(d_ss1, localStacks, d_ss2, stealStacks, config);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemcpy(h_ss1, d_ss1, sizeof(StackStats) * NBLOCKS, cudaMemcpyDeviceToHost);
    int leaves = 0, nodes = 0;
    for(int i = 0; i < NBLOCKS; ++i)
    {
        nodes += h_ss1[i].totalNodes;
        leaves += h_ss1[i].totalLeaves;
        printf("Block %d: nodes = %d, leaves = %d\n", i, h_ss1[i].totalNodes, h_ss1[i].totalLeaves);
    }
    printf("Total nodes = %d, total leaves = %d\n", nodes, leaves);
    return 0;
}
