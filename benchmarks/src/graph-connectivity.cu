/********************************************************************************************
 * Implementation of Parallel Graph Connectivity Computation
 *
 * Authored by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 *
 * Based on:
 * M. Sutton, T. Ben-Nun and A. Barak
 * Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)
 * Original Implementation: https://github.com/michaelsutton/afforest
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

__global__ void linkKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, int *graphComponents, int *bases, int *blockIds)
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
            int p1 = atomicAdd(&graphComponents[U], 0);
            int p2 = atomicAdd(&graphComponents[V], 0);

            while (p1 != p2)
            {
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
            int current = atomicAdd(&graphComponents[tid + my_base], 0);
            int parent = atomicAdd(&graphComponents[current], 0);

            while (current != parent)
            {
                int val = atomicCAS(&graphComponents[tid + my_base], current, parent);
                if(val == current)
                {
                    // Successful swap, continue compression
                    current = parent;
                    parent = atomicAdd(&graphComponents[current], 0);
                }
                else
                {
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

void outputValues(int *d_vertexComponent,  int V)
{
    int *h_vertexComponent = new int[V];
    errorCheck(cudaMemcpy(h_vertexComponent, d_vertexComponent, sizeof(int) * V, cudaMemcpyDeviceToHost), "copy vertexComponent DtH");
    ofstream out("connectivity-ans.txt");
    unordered_set<int> graphComponents; // Used to count number of components
    for(int i = 0; i < V; ++i)
    {
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
