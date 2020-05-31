/********************************************************************************************
 * Implementation of Parallel Graph Colouring
 *
 * Authored by: 
 * Aditya K Kamath, Indian Institute of Science
 *
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

/**************************************************
 *
 *               KERNEL FUNCTIONS
 *
 **************************************************/

__global__ void assignColorsKernel(int *head, int *tail, int *colorSet, int *vertexColor, int *VForbidden, int *TVForbidden, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0)
    {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last)
    {
        if(tid + my_base < n_t_last)
        {    
            if(vertexColor[tid + my_base] < 0)
                vertexColor[tid + my_base] = -1 * vertexColor[tid + my_base];
            else if(vertexColor[tid + my_base] == 0)
            {
                int allForbid = VForbidden[tid + my_base] | TVForbidden[tid + my_base];
                
                // Can probably be optimized using bitwise operations
                bool flag = false;
                for(int i = 0; i < 32; ++i)
                {
                    if((allForbid & (1 << i)) == 0)
                    {
                        vertexColor[tid + my_base] = i + 1;
                        flag = true;
                        break;
                    }
                }
                
                if(!flag)
                {
                    bool flag2 = false;
                    // Can probably be optimized using bitwise operations
                    for(int i = 0; i < 32; ++i)
                    {
                        if((VForbidden[tid + my_base] & (1<<i)) == 0)
                        {
                            flag2 = true;
                            break;
                        }
                    }
                    if(!flag2)
                    {
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

        if (tid == 0)
        {
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

__global__ void detectConflictsKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, int *colorSet, int *vertexColor, int *complete, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0)
    {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last)
    {
        if(tid + my_base < n_t_last)
        {    
    
            int U = edgeSetU[tid + my_base];
            int V = edgeSetV[tid + my_base];

            if(colorSet[U] == colorSet[V] && atomicAdd(&vertexColor[U], 0) == atomicAdd(&vertexColor[V], 0))
            {
                atomicExch(&vertexColor[U], 0);
            }

            if(atomicAdd(&vertexColor[U], 0) == 0 || atomicAdd(&vertexColor[V], 0) == 0)
            {
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

        if (tid == 0)
        {
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

__global__ void forbidColorsKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, int *colorSet, int *vertexColor, int *VForbidden, int *bases, int *blockIds)
{
    const int tid = threadIdx.x;
    int bid       = blockIdx.x;
    int *base     = &bases[bid];
    int *blockId  = &blockIds[bid];

    if(tid == 0)
    {
        *base = atomicAdd(&head[bid], NTHREADS);
        atomicExch_block(blockId, bid);
    }

    __syncthreads();

    int my_base = *base;
    int n_t_last = tail[bid];
    while(my_base < n_t_last)
    {
        if(tid + my_base < n_t_last)
        {    
            int U = edgeSetU[tid + my_base];
            int V = edgeSetV[tid + my_base];
            
            if(colorSet[U] == colorSet[V])
            {
                if(vertexColor[U] > 0 && vertexColor[V] == 0)
                {
                    atomicOr(&VForbidden[V], (1 << (vertexColor[U] - 1)));
                }

                if(vertexColor[U] == 0 && vertexColor[V] > 0)
                {
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

        if (tid == 0)
        {
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

/**************************************************
 *
 *                HOST FUNCTIONS
 *
 **************************************************/

void errorCheck(cudaError_t err, const char location[])
{
    if (err != cudaSuccess)
    {
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
    
    for(int i = 0; i < E; ++i)
    {
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
    if(value < size)
    {
        for(int i = 0; i < value; ++i)
        {
            h_head[i] = i;
            h_tail[i] = (i + 1);
        }
    }
    else
    {
        int portion = value / size;
        for(int i = 0; i < size; ++i)
        {
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
    for(int i = 0; i < V; ++i)
    {
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
    while(!(*h_complete))
    {
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, V);
        assignColorsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_colorSet, d_vertexColor, d_VForbidden, d_TVForbidden, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "assignColorsKernel");
        outputValues(d_colorSet, d_vertexColor, V);
        *h_complete = true;
        errorCheck(cudaMemcpy(d_complete, h_complete, sizeof(int), cudaMemcpyHostToDevice), "copy complete HtD");
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, E);
        detectConflictsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_edgeListU, d_edgeListV, d_colorSet, d_vertexColor, d_complete, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "detectConflictsKernel");
        errorCheck(cudaMemcpy(h_complete, d_complete, sizeof(int), cudaMemcpyDeviceToHost), "copy complete DtH");
        outputValues(d_colorSet, d_vertexColor, V);
        divideWork(h_head, h_tail, d_head, d_tail, NBLOCKS, E);
        forbidColorsKernel<<<dimGrid, dimBlock>>>
            (d_head, d_tail, d_edgeListU, d_edgeListV, d_colorSet, d_vertexColor, d_VForbidden, d_base, d_blockId);
        errorCheck(cudaDeviceSynchronize(), "forbidColorsKernel");
        outputValues(d_colorSet, d_vertexColor, V);
    }

    outputValues(d_colorSet, d_vertexColor, V);

    return 0;
}
