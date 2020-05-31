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


// @@@code  {Inter block data write outside critical section}   {cuda:bbwrwithoutlock}
__device__ int lock = 0;
__device__ int dummy = 0;

__global__ void kmain(volatile unsigned int *data)       // @@@{
{
    if(blockIdx.x == 0)
    {
        while(atomicCAS(&lock, 0, 1) != 0) {}
        __threadfence();
        dummy = data[0];
        __threadfence();
        atomicExch(&lock, 0);
        data[0] = 1;
    }
    else
    {
        while(atomicCAS(&lock, 0, 1) != 0) {}
        __threadfence();
        dummy = data[0];
        __threadfence();
        atomicExch(&lock, 0);
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

