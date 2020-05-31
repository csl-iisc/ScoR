#include <stdio.h>

#define NBLOCKS  2
#define TPERBLK  1

#define NTHREADS (NBLOCKS * TPERBLK)

void errCheck()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error %d: %s\n", err, cudaGetErrorString(err));
        exit(1);
    }
}


// @@@code  {inter block atomic race}   {cuda:bbatomic}
__device__ int flag = 0;
__device__ int dummy = 0;

__global__ void kmain(unsigned int *data)   // @@@{
{
    if(blockIdx.x == 0)
    {
        atomicExch_block(&data[0], 1);
    }
    else
    {
        atomicExch_block(&data[0], 2);
    }
}                                           // @@@}

int main() 
{
    unsigned int *d_data;
    cudaMalloc(&d_data, sizeof(unsigned int));
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    errCheck();
    return 0;
}

