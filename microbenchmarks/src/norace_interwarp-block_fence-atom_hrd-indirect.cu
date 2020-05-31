#include <stdio.h>

#define NBLOCKS  2
#define TPERBLK  33

#define NTHREADS (NBLOCKS * TPERBLK)

void errCheck()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error %d: %s\n", err, cudaGetErrorString(err));
        exit(1);
    }
}

__device__ int flag = 0;

__global__ void kmain(unsigned int *data) 
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        atomicExch_block(&data[0], 1);
        __threadfence_block();
        atomicExch(&flag, 1);
    }
    else if(blockIdx.x == 0 && threadIdx.x == 32)
    {
        while(atomicAdd(&flag, 0) != 1);
        atomicExch(&data[0], 2);
        __threadfence();
        atomicExch(&flag, 2);
    }
    else if(blockIdx.x == 1 && threadIdx.x == 0)
    {
        while(atomicAdd(&flag, 0) != 2);
        atomicExch_block(&data[0], 3);
        __threadfence_block();
        atomicExch(&flag, 3);
    }
    else if(blockIdx.x == 1 && threadIdx.x == 32)
    {
        while(atomicAdd(&flag, 0) != 3);
        atomicExch(&data[0], 3);
    }
}

int main() 
{
    unsigned int *d_data;
    cudaMalloc(&d_data, sizeof(unsigned int));
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    errCheck();
    return 0;
}

