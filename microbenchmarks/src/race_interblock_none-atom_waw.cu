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

// @@@code  {Inter block atomic vs non-atomic conflict}   {cuda:bbanconflict}
__device__ int flag = 0;
__device__ int dummy = 0;
__global__ void kmain(volatile unsigned int *data)       // @@@{
{
    if(blockIdx.x == 0)
    {
        atomicExch((unsigned int*)&data[0], 1);
    }
    else
    {
        data[0] = 2;
    }
}                                               // @@@}

int main() 
{
    unsigned int *d_data;
    cudaMalloc(&d_data, sizeof(unsigned int));
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    errCheck();
    return 0;
}

