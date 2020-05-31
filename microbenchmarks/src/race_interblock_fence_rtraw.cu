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


// @@@code  {Inter block acquire missing fence2}   {cuda:bbnofenceacq2}
__device__ int flag = 0;
__device__ int dummy = 0;

__global__ void kmain(volatile unsigned int *data)           // @@@{
{
    
    if(blockIdx.x == 0)
    {
        dummy = data[0];
        __threadfence();
        atomicExch(&flag, 1);
        dummy = data[0];
        
    }
    else
    {
        while(atomicExch(&flag, 0) == 0) {}
        data[0] = 1;
    }
}                                                   // @@@}

int main() 
{
    unsigned int *d_data;
    cudaMalloc(&d_data, sizeof(unsigned int));
    kmain<<<NBLOCKS,TPERBLK>>>(d_data);
    errCheck();
    return 0;
}

