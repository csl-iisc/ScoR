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

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#include "gcol_kernel.cuh"

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
 * Edges should have lowest index vertex first
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
void allocate(int **d_edgeListU,  int **d_edgeListV, int **d_colorSet, int **d_vertexColor, int **d_VForbidden, 
    int **d_TVForbidden, int **d_head, int **d_tail, int **d_complete, int **d_base, int **d_blockId, int V, int E)
{
    errorCheck(cudaMalloc((void**)d_edgeListU,   sizeof(int) * E), "allocate edgeListU");
    errorCheck(cudaMalloc((void**)d_edgeListV,   sizeof(int) * E), "allocate edgeListV");
    errorCheck(cudaMalloc((void**)d_colorSet,    sizeof(int) * V), "allocate colorSet");
    errorCheck(cudaMalloc((void**)d_vertexColor, sizeof(int) * V), "allocate vertexColor");
    errorCheck(cudaMalloc((void**)d_VForbidden,  sizeof(int) * V), "allocate VForbidden");
    errorCheck(cudaMalloc((void**)d_TVForbidden, sizeof(int) * V), "allocate TVForbidden");
    errorCheck(cudaMalloc((void**)d_complete,    sizeof(int)), "allocate complete");
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

void outputValues(int *d_colorSet, int *d_vertexColor,  int V)
{
    int *h_colorSet    = new int[V];
    int *h_vertexColor = new int[V];
    errorCheck(cudaMemcpy(h_colorSet,    d_colorSet,    sizeof(int) * V, cudaMemcpyDeviceToHost), "copy colorSet DtH");
    errorCheck(cudaMemcpy(h_vertexColor, d_vertexColor, sizeof(int) * V, cudaMemcpyDeviceToHost), "copy vertexColor DtH");
    int maxi = 0;
    ofstream out("color-ans.txt");
    for(int i = 0; i < V; ++i) {
        int value = h_colorSet[i] * 32 + h_vertexColor[i];
        out << value << "\n";
        if(value > maxi)
            maxi = value;
    }

    out << "Total colors: " << maxi << "\n";
    cout << "Total colors: " << maxi << "\n";
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
    int *h_complete = new int;
        
    // Declare device variables
    int *d_edgeListU,  *d_edgeListV;
    int *d_colorSet,   *d_vertexColor;
    int *d_VForbidden, *d_TVForbidden;
    int *d_head, *d_tail;
    int *d_base, *d_blockId;
    int *d_complete;

    allocate(&d_edgeListU, &d_edgeListV, &d_colorSet, &d_vertexColor,
        &d_VForbidden, &d_TVForbidden, &d_head, &d_tail, &d_complete, &d_base, &d_blockId, V, E);


    // Copy edge list to device
    errorCheck(cudaMemcpy(d_edgeListU, h_edgeListU, sizeof(int) * E, cudaMemcpyHostToDevice), "copy edgeListU HtD");
    errorCheck(cudaMemcpy(d_edgeListV, h_edgeListV, sizeof(int) * E, cudaMemcpyHostToDevice), "copy edgeListV HtD");
    errorCheck(cudaMemset(d_colorSet, 0, sizeof(int) * V), "memset colorSet");
    errorCheck(cudaMemset(d_vertexColor, 0, sizeof(int) * V), "memset vertexColor");
    errorCheck(cudaMemset(d_VForbidden, 0, sizeof(int) * V), "memset VForbidden");
    errorCheck(cudaMemset(d_TVForbidden, 0, sizeof(int) * V), "memset TVForbidden");
    
    dim3 dimGrid(NBLOCKS);
    dim3 dimBlock(NTHREADS);

    *h_complete = 0;
    // Begin coloring graph
    while(!(*h_complete)) {
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, V);
        assignColorsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_colorSet, d_vertexColor, d_VForbidden, d_TVForbidden, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "assignColorsKernel");
        *h_complete = true;
        errorCheck(cudaMemcpy(d_complete, h_complete, sizeof(int), cudaMemcpyHostToDevice), "copy complete HtD");
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, E);
        detectConflictsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_edgeListU, d_edgeListV, d_colorSet, d_vertexColor, d_complete, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "detectConflictsKernel");
        errorCheck(cudaMemcpy(h_complete, d_complete, sizeof(int), cudaMemcpyDeviceToHost), "copy complete DtH");
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, E);
        forbidColorsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_edgeListU, d_edgeListV, d_colorSet, d_vertexColor, d_VForbidden, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "forbidColorsKernel");
    }

    outputValues(d_colorSet, d_vertexColor, V);

    return 0;
}
