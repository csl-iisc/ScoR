 /********************************************************************************************
 * Implementation of Unbalanced Tree Search
 *
 * Authored by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 * Based on:
 * S. Olivier, J. Huan, J. Liu, J. Prins, J. Dinan, P. Sadayappan, C. Tseng
 * UTS: An Unbalanced Tree Search Benchmarks
 * 2006 International Workshop on Languages and Compilers for Parallel Computing (LCPC)
 * 
 ********************************************************************************************/

#define WARP_SIZE (NTHREADS < 32 ? NTHREADS : 32)
#define MAXNUMCHILDREN    100  // cap on children

#define MAXSTACKDEPTH 4000
#define LOCAL_DEPTH (NTHREADS * 4)

#define SYNC_WARP

struct Node
{
  // Height, numChildren, seed, num
  // Stored as char array for better write-efficiency
  unsigned char param[4];
};

/* stack of nodes */
struct StackStats
{
  volatile int stackSize;     /* total space avail (in number of elements) */
  volatile int workAvail;     /* total elements */
  volatile int top;           /* index of stack top */
  int locked;                 /* used to lock stack */
  int totalNodes;
  int totalLeaves;
};

struct Config
{
    int maxHeight;
    int avgChildren;
};

#include <stdio.h>

__device__ int getRandNum(Node *parent, Config config)
{
    return (((unsigned int)parent->param[2] * (unsigned int)parent->param[3] + 5) % 255);
}

__device__ void genChildren(Node *parent, StackStats *localStackStats, Node *localStacks, StackStats *stealStackStats, Node *stealStacks, Config config) 
{  
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int wtid = tid % WARP_SIZE;
  
  int num = getRandNum(parent, config);
  int numChildren;
  
  if(parent->param[0] + 1 >= config.maxHeight)
    numChildren = 0;
  else
    numChildren = num % (config.avgChildren * 2 - 1) + 1;
    
  Node child;
  child.param[0] = parent->param[0] + 1;
  child.param[1] = 255;
  child.param[2] = num;
  int ctr = 1;

  // construct children and push onto stack  
  SYNC_WARP;
  while(wtid == 0 && atomicCAS_block(&localStackStats[bid].locked, 0, 1) != 0);
  __threadfence_block();
  SYNC_WARP;
  int extra = 0;
  if(parent->param[1] == 255)
  {
    parent->param[1] = numChildren;
    atomicAdd_block(&localStackStats[bid].totalNodes, numChildren);
    if(numChildren == 0)
      atomicAdd_block(&localStackStats[bid].totalLeaves, 1);
    if(numChildren > 0)
    {
        int amount = atomicAdd((int *)&localStackStats[bid].workAvail, numChildren) + numChildren;
        if(amount > LOCAL_DEPTH)
        {
          extra = min(amount - LOCAL_DEPTH, numChildren);
          numChildren -= extra;
        }
        if(numChildren > 0)
        {
          int start = atomicAdd_block((int *)&localStackStats[bid].top, -1 * numChildren) - 1;
          for(int i = 0; i < numChildren; ++i)
          {
            if(start < LOCAL_DEPTH * bid)
              start += LOCAL_DEPTH;
            child.param[3] = ctr;
            ++ctr;
            atomicExch_block((int *)&localStacks[start].param, *(int *)child.param);
            --start;
          }
        }
    }
  }
  SYNC_WARP;
  if(wtid == 0 && atomicAdd_block((int *)&localStackStats[bid].top, 0) < LOCAL_DEPTH * bid)
  {
    atomicAdd_block((int *)&localStackStats[bid].top, LOCAL_DEPTH);
  }
  atomicExch((int*)&localStackStats[bid].workAvail, min(atomicAdd((int*)&localStackStats[bid].workAvail, 0), LOCAL_DEPTH));
  SYNC_WARP;
  if(wtid == 0)
  {
    __threadfence_block();
    atomicExch_block(&localStackStats[bid].locked, 0);
  }
  SYNC_WARP;
#ifdef RACEY
  while(wtid == 0 && atomicCAS_block(&stealStackStats[bid].locked, 0, 1) != 0);
#else
  while(wtid == 0 && atomicCAS(&stealStackStats[bid].locked, 0, 1) != 0);
#endif
  __threadfence();
  SYNC_WARP;
  if(extra > 0)
  {
#ifdef RACEY
    int amount = atomicAdd_block((int *)&stealStackStats[bid].workAvail, extra) + extra;
#else
    int amount = atomicAdd((int *)&stealStackStats[bid].workAvail, extra) + extra;
#endif
    int start = atomicSub((int *)&stealStackStats[bid].top, extra) - 1;
    for (int i = 0; i < extra; ++i)
    {
      if(start < MAXSTACKDEPTH * bid)
        start += MAXSTACKDEPTH;

      child.param[3] = ctr;
      ++ctr;
      atomicExch((int *)&stealStacks[start].param, *(int *)child.param);
      --start;
    }
  }
  SYNC_WARP;
  if(wtid == 0 && atomicAdd((int *)&stealStackStats[bid].top, 0) < MAXSTACKDEPTH * bid)
  {
    atomicAdd((int *)&stealStackStats[bid].top, MAXSTACKDEPTH);
  }
  SYNC_WARP;
  if(wtid == 0)
  {
      __threadfence();
#ifdef RACEY
      atomicExch_block(&stealStackStats[bid].locked, 0);
#else
      atomicExch(&stealStackStats[bid].locked, 0);
#endif
  }
  SYNC_WARP;
}

__device__ bool checkWork(StackStats *localStacks, StackStats *stealStacks)
{
    __shared__ char done[(NTHREADS + WARP_SIZE - 1) / WARP_SIZE];
    const int wtid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    if(wtid == 0)
    {
        done[wid] = 1;
        for(int i = 0; i < gridDim.x; ++i)
            if(atomicAdd((int*)&localStacks[i].workAvail, 0) > 0 || atomicAdd((int*)&stealStacks[i].workAvail, 0) > 0)
            {
                done[wid] = 0;
                break;
            }
    }
    return done[wid];
}

__global__ void parTreeSearch(StackStats *localStackStats, Node *localStacks, StackStats *stealStackStats, Node *stealStacks, Config config) 
{
    int done = 0;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wtid = tid % WARP_SIZE;
    Node parent;
    parent.param[0] = 0;
    int i = bid;
    /* tree search */
    while (done == 0) 
    {
      parent.param[1] = 0;
      // Have warp acquire lock for local stack
      SYNC_WARP;
      while(wtid == 0 && atomicCAS_block(&localStackStats[bid].locked, 0, 1) != 0);
      __threadfence_block();
      SYNC_WARP;
      if (atomicAdd((int*)&localStackStats[bid].workAvail, 0) > wtid) 
      {
          *(unsigned int *)(parent.param) = atomicAdd_block((int *)&localStacks[(localStackStats[bid].top + wtid) % LOCAL_DEPTH + LOCAL_DEPTH * bid].param, 0);
          if(wtid == 0)
          {
              localStackStats[bid].top = (localStackStats[bid].top + min(WARP_SIZE, atomicAdd((int*)&localStackStats[bid].workAvail, 0))) % LOCAL_DEPTH + LOCAL_DEPTH * bid;
              int val = atomicSub((int*)&localStackStats[bid].workAvail, WARP_SIZE);
              if(val - WARP_SIZE < 0)
                  atomicExch((int*)&localStackStats[bid].workAvail, 0);
          }
      }
      SYNC_WARP;
      if(wtid == 0)
      {
        __threadfence_block();
        atomicExch_block(&localStackStats[bid].locked, 0);
      }
      SYNC_WARP;
      genChildren(&parent, localStackStats, localStacks, stealStackStats, stealStacks, config);
      // Check steal stacks now
      {
          int blk = i;
          i = ((i + 1) % NBLOCKS);
          parent.param[1] = 0;
          SYNC_WARP;
#ifdef RACEY
          while(wtid == 0 && atomicCAS_block(&stealStackStats[blk].locked, 0, 1) != 0);
#else
          while(wtid == 0 && atomicCAS(&stealStackStats[blk].locked, 0, 1) != 0);
#endif
          __threadfence();
          SYNC_WARP;
          if(atomicAdd((int*)&stealStackStats[blk].workAvail, 0) > wtid)
          {
              *(unsigned int *)(parent.param) = atomicAdd((int *)&stealStacks[(stealStackStats[blk].top + wtid) % MAXSTACKDEPTH + MAXSTACKDEPTH * blk].param, 0);
              if(wtid == 0)
              {
                  stealStackStats[blk].top = (stealStackStats[blk].top + min(WARP_SIZE, atomicAdd((int*)&stealStackStats[blk].workAvail, 0))) % MAXSTACKDEPTH + MAXSTACKDEPTH * blk;
                  int val = atomicSub((int*)&stealStackStats[blk].workAvail, WARP_SIZE);
                  if(val - WARP_SIZE < 0)
#ifdef RACEY
                      atomicExch_block((int*)&stealStackStats[blk].workAvail, 0);
#else
                      atomicExch((int*)&stealStackStats[blk].workAvail, 0);
#endif
              }
          }
          SYNC_WARP;
          if(wtid == 0)
          {
              __threadfence();
#ifdef RACEY
              atomicExch_block(&stealStackStats[blk].locked, 0);
#else
              atomicExch(&stealStackStats[blk].locked, 0);
#endif
          }
          SYNC_WARP;
          genChildren(&parent, localStackStats, localStacks, stealStackStats, stealStacks, config);
      }
      
      done = checkWork(localStackStats, stealStackStats);
    }
  
  /* tree search complete ! */
}


int main(int argc, char *argv[])
{
    //parseParams(argc, argv);
    StackStats *d_ss1, *d_ss2, *h_ss1, *h_ss2;
    Node *localStacks, *stealStacks;
    
    Config config;
    scanf("%d %d", &config.maxHeight, &config.avgChildren);
    
    cudaMalloc((void **)&d_ss1, sizeof(StackStats) * NBLOCKS);
    cudaMalloc((void **)&d_ss2, sizeof(StackStats) * NBLOCKS);
    cudaMalloc((void **)&localStacks, sizeof(Node) * (NBLOCKS * LOCAL_DEPTH));
    cudaMalloc((void **)&stealStacks, sizeof(Node) * (NBLOCKS * MAXSTACKDEPTH));
    
    h_ss1 = (StackStats *)malloc(sizeof(StackStats) * NBLOCKS);
    for(int i = 0; i < NBLOCKS; ++i)
    {
        h_ss1[i].stackSize = LOCAL_DEPTH;
        h_ss1[i].workAvail = 1;
        h_ss1[i].top = LOCAL_DEPTH * i;
        h_ss1[i].totalNodes = 1;
        h_ss1[i].totalLeaves = 0;
        h_ss1[i].locked = 0;
    }
    
    h_ss2 = (StackStats *)malloc(sizeof(StackStats) * NBLOCKS);
    for(int i = 0; i < NBLOCKS; ++i)
    {
        h_ss2[i].stackSize = MAXSTACKDEPTH;
        h_ss2[i].workAvail = 0;
        h_ss2[i].top = MAXSTACKDEPTH * i;
        h_ss2[i].totalNodes = 0;
        h_ss2[i].totalLeaves = 0;
        h_ss2[i].locked = 0;
    }
    
    cudaMemcpy(d_ss1, h_ss1, sizeof(StackStats) * (NBLOCKS), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ss2, h_ss2, sizeof(StackStats) * (NBLOCKS), cudaMemcpyHostToDevice);
    
    Node root;
    root.param[0] = 0;
    root.param[1] = 255;
    int seed;
    scanf("%d", &seed);
    root.param[2] = (seed % 255);
    for(int i = 0; i < NBLOCKS; ++i)
    {
        root.param[3] = (i + 1) % 255;
        cudaMemcpy(&localStacks[LOCAL_DEPTH * i], &root, sizeof(Node), cudaMemcpyHostToDevice);
    }
    parTreeSearch<<<NBLOCKS, NTHREADS>>>(d_ss1, localStacks, d_ss2, stealStacks, config);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaMemcpy(h_ss1, d_ss1, sizeof(StackStats) * NBLOCKS, cudaMemcpyDeviceToHost);
    int leaves = 0, nodes = 0;
    for(int i = 0; i < NBLOCKS; ++i)
    {
        nodes += h_ss1[i].totalNodes;
        leaves += h_ss1[i].totalLeaves;
        printf("Block %d: nodes = %d, leaves = %d\n", i, h_ss1[i].totalNodes, h_ss1[i].totalLeaves);
    }
    printf("Total nodes = %d, total leaves = %d\n", nodes, leaves);
    return 0;
}
