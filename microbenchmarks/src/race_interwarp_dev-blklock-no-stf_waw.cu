#include <stdio.h>

#define NBLOCKS  1
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

__device__ int lock = 0;

__global__ void kmain(volatile unsigned int *data) 
{
    if(threadIdx.x == 0)
    {
        while(atomicCAS(&lock, 0, 1) != 0) {}
        //__threadfence(); -- Missing
        data[0] = 1;
        __threadfence();
        atomicExch(&lock, 0);
    }
    else if(threadIdx.x == 32)
    {
        while(atomicCAS_block(&lock, 0, 1) != 0) {}
        __threadfence_block();
        data[0] = 2;
        __threadfence_block();
        atomicExch_block(&lock, 0);
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

