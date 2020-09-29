/********************************************************************************************
 * Copyright (c) 2020 Indian Institute of Science
 * All rights reserved.
 *
 * Developed by:    Aditya K Kamath
 *                  Computer Systems Lab
 *                  Indian Institute of Science
 *                  https://csl.csa.iisc.ac.in/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of Computer Systems Lab, Indian Institute of Science, 
 *        nor the names of its contributors may be used to endorse or promote products 
 *        derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ********************************************************************************************/

 /********************************************************************************************
 * Implementation of Unbalanced Tree Search
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

#define MAX_CHAR 255

// Warp-level sync needed for Independent Thread Scheduling
#if __CUDA_ARCH__ >= 700
#define SYNC_WARP __syncwarp();
#else
#define SYNC_WARP
#endif

struct Node
{
  // Height, numChildren, seed, num
  // Stored as char array for better write-efficiency
  unsigned char param[4];
};

typedef enum {
    HEIGHT   = 0,
    CHILDREN = 1,
    SEED     = 2,
    NUMBER   = 3,
} node_params;

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

__device__ 
int getRandNum(Node *parent, Config config);

__device__ 
void genChildren(Node *parent, StackStats *localStackStats, Node *localStacks, 
    StackStats *stealStackStats, Node *stealStacks, Config config);

__device__ 
bool checkWork(StackStats *localStacks, StackStats *stealStacks);

__global__ 
void parTreeSearch(StackStats *localStackStats, Node *localStacks, 
    StackStats *stealStackStats, Node *stealStacks, Config config);
