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

__device__ int flag = 0;

__global__ void kmain(volatile unsigned int *data) 
{
    if(threadIdx.x == 0)
    {
        data[0] = 1;
        __threadfence();
        atomicExch(&flag, 1);
    }
    else if(threadIdx.x == 32)
    {
        while(atomicExch(&flag, 0) == 0) {}
        int a = data[0];
        atomicExch(&flag, a);
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

