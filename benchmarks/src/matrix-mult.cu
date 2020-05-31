 /********************************************************************************************
 * Implementation of Matrix Multiplication using Threadfence Locks
 *
 * Authored by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 * Each block contains a subpart of a row of the input matrix.
 *
 * Use lockset for locking mechanism between blocks/threads.
 *
 ********************************************************************************************/

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#define WARP_SIZE (NTHREADS < 32 ? NTHREADS : 32)

/***************************************************
 *
 *               KERNEL FUNCTIONS
 *
 ***************************************************/

#define datatype int

// Input: Matrix A, Matrix B, Result Matrix C, # rows of A (= # rows of C), 
//        # columns of A (= # rows of B), # columns of B (= # columns of C)
// All in row major format
__global__ void matMultKernel(datatype *A, datatype *B, volatile datatype *C, datatype rA, datatype cA, datatype cB, int *sh_locks, volatile datatype *tempCs, volatile int *gl_lock)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Allow each warp its own lock for better parallelism
    int &sh_lock = sh_locks[bid * WARP_SIZE + (tid % WARP_SIZE)];
    volatile datatype &tempC = tempCs[bid * WARP_SIZE + (tid % WARP_SIZE)];

    int offset = 0;

    if(tid == 0)
        sh_lock = 0;
    __syncthreads();
    
    // Repeat until entire matrix has been covered
    while(offset / ((cA + blockDim.x - 1) / blockDim.x) < rA)
    {
        int row = (blockIdx.x + offset) / ((cA + blockDim.x - 1) / blockDim.x);
        // Iterate over columns in matrix B
        for(int i = 0; i < (cB + WARP_SIZE - 1) / WARP_SIZE; ++i)
        {
            // Initialize shared variables
            if(tid / WARP_SIZE == 0)
            {
                tempC = 0;
            }
            __syncthreads();
            for(int j = 0; j < WARP_SIZE; ++j)
            {
                int col = ((blockIdx.x + offset) % ((cA + blockDim.x - 1) / blockDim.x)) * blockDim.x + (tid / WARP_SIZE) * WARP_SIZE + j;
                datatype t = 0;
                if(row < rA && col < cA && (i *  WARP_SIZE) + (tid % WARP_SIZE) < cB)
                {
                    // Perform multiplication
                    t = A[col + row * cA] * B[(i *  WARP_SIZE) + (tid % WARP_SIZE) + col * cB];
                }
                
                // Have each thread update shared variable with value
                bool success = false;
                do{
                    if(atomicCAS_block(&sh_lock, 0, 1) == 0)
                    {
#ifdef RACEY
#else
                        __threadfence_block();
#endif
                        tempC += t;
#ifdef RACEY
#else
                        __threadfence_block();
#endif
                        atomicExch_block(&sh_lock, 0);
                        success = true;
                    }
                }while(!success);
            }
            __syncthreads();
            
            // Update the result array with value
            if(tid / WARP_SIZE == 0 && row < rA && i * WARP_SIZE + tid < cB)
            {
                bool successful = false;
                do{
#ifdef RACEY
                    // 2 races: One on gl_lock due to block atomic, other on C, due to block-scope locking
                    if(0 == atomicCAS_block((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % 1024], 0, 1))
                    {
                        __threadfence_block();
#else
                    if(0 == atomicCAS((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % 1024], 0, 1))
                    {
                        __threadfence();
#endif
                        C[i * WARP_SIZE + tid + row * cB] += tempC;__threadfence();
                        atomicExch((int*)&gl_lock[(i * WARP_SIZE + tid + row * cB) % 1024], 0);
                        successful = true;
                    }
                }while(!successful);
            }
        }
        
        offset += gridDim.x;
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

void input(int &rA, int &cA, int &cB, datatype **matA, datatype **matB, datatype **matC)
{
    cin >> rA >> cA >> cB;
    *matA = new datatype[rA * cA];
    *matB = new datatype[cA * cB];
    *matC = new datatype[rA * cB];

    for(int i = 0; i < rA * cA; ++i)
        cin >> (*matA)[i];
    for(int i = 0; i < cA * cB; ++i)
        cin >> (*matB)[i];
    for(int i = 0; i < rA * cB; ++i)
        (*matC)[i] = 0;
}

void outputValues(int rows, int cols, datatype *matC)
{
    ofstream out("matrix-ans.txt");
    for(int i = 0; i < rows * cols; ++i)
    {
        out << matC[i] << " ";
        if(i % cols == cols - 1)
            out << "\n";
    }
}

int main()
{
    int rA, cA, cB;
    // Declare host variables
    datatype *h_matA, *h_matB, *h_matC;
    // Input data
    input(rA, cA, cB, &h_matA, &h_matB, &h_matC);
    // Declare device variables
    datatype *d_matA, *d_matB, *d_matC, *d_sh_lock, *d_tempC, *d_g_lock;

    errorCheck(cudaMalloc(&d_matA, sizeof(datatype) * rA * cA), "Malloc device matA");
    errorCheck(cudaMalloc(&d_matB, sizeof(datatype) * cA * cB), "Malloc device matB");
    errorCheck(cudaMalloc(&d_matC, sizeof(datatype) * rA * cB), "Malloc device matC");
    errorCheck(cudaMalloc(&d_sh_lock, sizeof(int) * NBLOCKS * WARP_SIZE), "Malloc device sh_lock");
    errorCheck(cudaMalloc(&d_g_lock, sizeof(int) * 1024), "Malloc device global lock");
    errorCheck(cudaMalloc(&d_tempC, sizeof(datatype) * NBLOCKS * WARP_SIZE), "Malloc device tempC");

    
    errorCheck(cudaMemset(d_g_lock, 0, sizeof(int) * 1024), "Memset g_lock");
    errorCheck(cudaMemcpy(d_matA, h_matA, sizeof(datatype) * rA * cA, cudaMemcpyHostToDevice), "Memcpy device matA");
    errorCheck(cudaMemcpy(d_matB, h_matB, sizeof(datatype) * cA * cB, cudaMemcpyHostToDevice), "Memcpy device matB");
    errorCheck(cudaMemcpy(d_matC, h_matC, sizeof(datatype) * rA * cB, cudaMemcpyHostToDevice), "Memcpy device matC");

    matMultKernel<<<NBLOCKS, NTHREADS>>>(d_matA, d_matB, d_matC, rA, cA, cB, d_sh_lock, d_tempC, d_g_lock);

    errorCheck(cudaMemcpy(h_matC, d_matC, sizeof(datatype) * rA * cB, cudaMemcpyDeviceToHost), "Memcpy host matC");
    outputValues(rA, cB, h_matC);
    return 0;
}
