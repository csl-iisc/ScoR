 /********************************************************************************************
 * Implementation of Rule 110 Cellular Automaton
 *
 * Authored by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 ********************************************************************************************/

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#define WARP_SIZE (NTHREADS < 32 ? NTHREADS : 32)
#define DATA int

__device__ volatile int gl_lock = 0;
/***************************************************
 *
 *               DEVICE FUNCTIONS
 *
 ***************************************************/
__device__ bool isBorder()
{
    return threadIdx.x == 0 || threadIdx.x == blockDim.x - 1;
}

/*
 * From Wikipedia:
 * Current pattern           111 110 101 100 011 010 001 000
 * New state for center cell  0   1   1   0   1   1   1   0
 */
/***************************************************
 *
 *               KERNEL FUNCTIONS
 *
 ***************************************************/

__global__ void rule110Kernel(DATA *arr, volatile DATA *copy, int *comp, const int size, int step)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int btid = threadIdx.x;
    int offset = 0;
    while(offset < size)
    {
        int ind = tid + offset;
        if(ind < size)
        {
            // Copy array value into a copy
            copy[ind] = arr[ind];
        }
        offset += NBLOCKS * NTHREADS;
    }
    // Threadfence and mark complete
    if(isBorder())
    {
#ifdef RACEY
        __threadfence_block();
#else
        __threadfence();
#endif
        atomicExch(&comp[tid], step);
    }
    else
    {
        __threadfence_block();
        atomicExch_block(&comp[tid], step);
    }
    
    // Use device scope for border neighbors, and block for non-border
    if(btid - 1 == 0 || btid == 0)
    {
        while(step != atomicAdd(&comp[(NBLOCKS * NTHREADS + tid - 1) % (NBLOCKS * NTHREADS)], 0));
        __threadfence();
    }
    else
    {
        while(step != atomicAdd_block(&comp[tid - 1], 0));
        __threadfence_block();
    }
    
    offset = 0;
    while(offset < size)
    {
        int ind = tid + offset;
        
        if(ind < size)
        {
            if(ind > 0)
                arr[ind] = copy[ind - 1];
            else
                arr[ind] = 0;
        }
        offset += NBLOCKS * NTHREADS;
    }
    
    if(btid + 1 == NTHREADS || btid + 1 == NTHREADS - 1)
    {
#ifdef RACEY
        while(step != atomicAdd_block(&comp[(NBLOCKS * NTHREADS + tid + 1) % (NBLOCKS * NTHREADS)], 0));
#else
        while(step != atomicAdd(&comp[(NBLOCKS * NTHREADS + tid + 1) % (NBLOCKS * NTHREADS)], 0));
#endif
        __threadfence();
    }
    else
    {
        while(step != atomicAdd_block(&comp[tid + 1], 0));
        __threadfence_block();
    }
    
    // Calculate new value of array and update
    offset = 0;
    while(offset < size)
    {
        int ind = tid + offset;
        if(ind < size)
        {
            int val = arr[ind];
            val <<= 1;
            if(ind < size)
                val += copy[ind];
            val <<= 1;

            if(ind + 1 < size)
                val += copy[ind + 1];
            // Update value based on neighbors
            switch(val)
            {
                case 7:
                case 4:
                case 0:
                    val = 0;
                break;
                case 6:
                case 5:
                case 3:
                case 2:
                case 1:
                    val = 1;
                break;                        
            }
            // Calculate new value and store
            arr[ind] = val;
        }
        offset += NBLOCKS * NTHREADS;
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
 *
 *
 *
 *
 *
 *
 *
 *************************************************/

void input(int &size, int &steps, DATA **arr)
{
    cin >> size >> steps;
    *arr = new DATA[size];

    for(int i = 0; i < size; ++i)
        cin >> (*arr)[i];
}

void outputValues(int size, DATA *arr)
{
    ofstream out("rule110-ans.txt");
    for(int i = 0; i < size; ++i)
        out << arr[i] << " ";
    out << "\n";
}

int main()
{
    int size, steps;
    // Declare host variables
    DATA *h_arr;
    // Input data
    input(size, steps, &h_arr);
    // Declare device variables
    DATA *d_arr, *d_copy;
    int *d_comp;

    errorCheck(cudaMalloc(&d_arr, sizeof(DATA) * size), "Malloc device arr");
    errorCheck(cudaMalloc(&d_copy, sizeof(DATA) * size), "Malloc device copy");
    errorCheck(cudaMalloc(&d_comp, sizeof(int) * NBLOCKS * NTHREADS), "Malloc device complete");
    errorCheck(cudaMemcpy(d_arr, h_arr, sizeof(DATA) * size, cudaMemcpyHostToDevice), "Memcpy device arr");
    errorCheck(cudaMemset(d_comp, 0, sizeof(int) * size), "Memset complete");
    for(int i = 0; i < steps; ++i)
        rule110Kernel<<<NBLOCKS, NTHREADS>>>(d_arr, d_copy, d_comp, size, i + 1);
    errorCheck(cudaMemcpy(h_arr, d_arr, sizeof(DATA) * size, cudaMemcpyDeviceToHost), "Memcpy host arr");
    outputValues(size, h_arr);
    return 0;
}
