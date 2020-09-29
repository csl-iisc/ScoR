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

#include <iostream>
#include <unordered_set>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#include "gcon_kernel.cuh"

/**************************************************
 *
 *                HOST FUNCTIONS
 *
 **************************************************/

void errorCheck(cudaError_t err, const char location[])
{
    if (err != cudaSuccess) {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "; at " << location << "\n";
        exit(1);
    }
}

/**************************************************
 *              INPUT DESCRIPTION
 *
 * V = Number of vertices
 * E = Number of edges
 *
 * For all i < E
 * where u = edgeListU[i], and v = edgeListV[i]
 * (u, v) is an edge in the graph
 *
 *************************************************/

void input(int &V, int &E, int **edgeListU, int **edgeListV)
{    
    cin >> V >> E;
    (*edgeListU) = new int[E];
    (*edgeListV) = new int[E];
    
    for(int i = 0; i < E; ++i) {
        cin >> (*edgeListU)[i] >> (*edgeListV)[i];
    }
}

// Allocate sufficient space on GPU memory
void allocate(int **d_edgeListU,  int **d_edgeListV, int **d_vertexComponent, int **d_head, int **d_tail, int **d_base, int **d_blockId, int V, int E)
{
    errorCheck(cudaMalloc((void**)d_edgeListU, sizeof(int) * E), "allocate edgeListU");
    errorCheck(cudaMalloc((void**)d_edgeListV, sizeof(int) * E), "allocate edgeListV");
    errorCheck(cudaMalloc((void**)d_vertexComponent, sizeof(int) * V), "allocate vertexColor");
    errorCheck(cudaMalloc((void**)d_head, sizeof(int) * NBLOCKS), "allocate head");
    errorCheck(cudaMalloc((void**)d_tail, sizeof(int) * NBLOCKS), "allocate tail");
    errorCheck(cudaMalloc((void**)d_base, sizeof(int) * NBLOCKS), "allocate base");
    errorCheck(cudaMalloc((void**)d_blockId, sizeof(int) * NBLOCKS), "allocate blockId");
}

// Divide the edges/vertices amongst the blocks
void divideWork(int *h_head, int *h_tail, int *d_head, int *d_tail, int size, int value)
{
    if(value < size) {
        for(int i = 0; i < value; ++i) {
            h_head[i] = i;
            h_tail[i] = (i + 1);
        }
    }
    else {
        int portion = value / size;
        for(int i = 0; i < size; ++i) {
            h_head[i] = i * portion;
            h_tail[i] = (i + 1) * portion;
        }
        h_tail[size - 1] = value;
    }
    errorCheck(cudaMemcpy(d_head, h_head, sizeof(int) * NBLOCKS, cudaMemcpyHostToDevice), "copy head HtD1");
    errorCheck(cudaMemcpy(d_tail, h_tail, sizeof(int) * NBLOCKS, cudaMemcpyHostToDevice), "copy tail HtD1");
}

void outputValues(int *d_vertexComponent,  int V)
{
    int *h_vertexComponent = new int[V];
    errorCheck(cudaMemcpy(h_vertexComponent, d_vertexComponent, sizeof(int) * V, cudaMemcpyDeviceToHost), "copy vertexComponent DtH");
    ofstream out("connectivity-ans.txt");
    unordered_set<int> graphComponents; // Used to count number of components
    for(int i = 0; i < V; ++i) {
        out << h_vertexComponent[i] << "\n";
        graphComponents.emplace(h_vertexComponent[i]);
    }

    out << "Total components: " << graphComponents.size() << "\n";
    cout << "Total components: " << graphComponents.size() << "\n";
}

int main()
{
    // Declare and input graph details
    int V, E;
    int *h_edgeListU, *h_edgeListV;
    
    input(V, E, &h_edgeListU, &h_edgeListV);
    
    // Declare other host variables

    int *h_head  = new int[NBLOCKS];
    int *h_tail  = new int[NBLOCKS];
        
    // Declare device variables
    int *d_edgeListU,  *d_edgeListV;
    int *d_vertexComponent;
    int *d_head, *d_tail;
    int *d_base, *d_blockId;

    allocate(&d_edgeListU, &d_edgeListV, &d_vertexComponent, &d_head, &d_tail, &d_base, &d_blockId, V, E);

    // Copy edge list to device
    errorCheck(cudaMemcpy(d_edgeListU, h_edgeListU, sizeof(int) * E, cudaMemcpyHostToDevice), "copy edgeListU HtD");
    errorCheck(cudaMemcpy(d_edgeListV, h_edgeListV, sizeof(int) * E, cudaMemcpyHostToDevice), "copy edgeListV HtD");
    
    dim3 dimGrid(NBLOCKS);
    dim3 dimBlock(NTHREADS);

    // Begin connecting graph
    divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, V);
    initKernel<<<dimGrid, dimBlock>>>(d_head, d_tail, d_vertexComponent, d_base, d_blockId);
    errorCheck(cudaDeviceSynchronize(), "initKernel");
    
    divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, E);
    linkKernel<<<dimGrid, dimBlock>>>(d_head, d_tail, d_edgeListU, d_edgeListV, d_vertexComponent, d_base, d_blockId);
    errorCheck(cudaDeviceSynchronize(), "linkKernel");

    divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, V);
    compressKernel<<<dimGrid, dimBlock>>>(d_head, d_tail, d_vertexComponent, d_base, d_blockId);
    errorCheck(cudaDeviceSynchronize(), "compressKernel");    
    
    outputValues(d_vertexComponent, V);

    return 0;
}
