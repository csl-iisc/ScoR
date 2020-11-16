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

#include "uts_kernel.cuh"

// Random hash function, ideally should be SHA1
__device__ int getRandNum(Node *parent, Config config)
{
    return (((unsigned int)parent->param[SEED] * (unsigned int)parent->param[NUMBER] + 5) % MAX_CHAR);
}

// Generate children of a given parent and insert into appropriate stacks
__device__ void genChildren(Node *parent, StackStats *localStackStats, Node *localStacks, 
    StackStats *stealStackStats, Node *stealStacks, Config config) 
{  
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wtid = tid % WARP_SIZE;
  
    int num = getRandNum(parent, config);
    int numChildren;
  
    if(parent->param[HEIGHT] + 1 >= config.maxHeight)
        numChildren = 0;
    else
        numChildren = num % (config.avgChildren * 2 - 1) + 1;
    
    Node child;
    child.param[HEIGHT] = parent->param[0] + 1;
    child.param[CHILDREN] = MAX_CHAR;
    child.param[SEED] = num;
    int ctr = 1;

    // construct children and push onto stack  
    SYNC_WARP;
    while(wtid == 0 && atomicCAS_block(&localStackStats[bid].locked, 0, 1) != 0);
    __threadfence_block();
    SYNC_WARP;

    int extra = 0;
    if(parent->param[CHILDREN] == MAX_CHAR) {
        parent->param[CHILDREN] = numChildren;
        atomicAdd_block(&localStackStats[bid].totalNodes, numChildren);
    
        if(numChildren == 0)
            atomicAdd_block(&localStackStats[bid].totalLeaves, 1);
    
        if(numChildren > 0) {
            int amount = atomicAdd((int *)&localStackStats[bid].workAvail, 
                numChildren) + numChildren;
            if(amount > LOCAL_DEPTH) {
                extra = min(amount - LOCAL_DEPTH, numChildren);
                numChildren -= extra;
            }
            if(numChildren > 0) {
                int start = atomicAdd_block((int *)&localStackStats[bid].top, -1 * numChildren) - 1;
                for(int i = 0; i < numChildren; ++i) {
                    if(start < LOCAL_DEPTH * bid)
                        start += LOCAL_DEPTH;
                    child.param[NUMBER] = ctr;
                    ++ctr;
                    atomicExch_block((int *)&localStacks[start].param, *(int *)child.param);
                    --start;
                }
            }
        }
    }
    
    SYNC_WARP;
    if(wtid == 0 && atomicAdd_block((int *)&localStackStats[bid].top, 0) < LOCAL_DEPTH * bid) {
        atomicAdd_block((int *)&localStackStats[bid].top, LOCAL_DEPTH);
    }
    atomicExch((int*)&localStackStats[bid].workAvail, 
    min(atomicAdd((int*)&localStackStats[bid].workAvail, 0), LOCAL_DEPTH));
    SYNC_WARP;
    if(wtid == 0) {
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
  
    if(extra > 0) {
#ifdef RACEY
        int amount = atomicAdd_block((int *)&stealStackStats[bid].workAvail, extra) + extra;
#else
        int amount = atomicAdd((int *)&stealStackStats[bid].workAvail, extra) + extra;
#endif
        int start = atomicSub((int *)&stealStackStats[bid].top, extra) - 1;
        for (int i = 0; i < extra; ++i) {
            if(start < MAXSTACKDEPTH * bid)
                start += MAXSTACKDEPTH;

            child.param[NUMBER] = ctr;
            ++ctr;
            atomicExch((int *)&stealStacks[start].param, *(int *)child.param);
            --start;
        }
    }
  
    SYNC_WARP;
    if(wtid == 0 && atomicAdd((int *)&stealStackStats[bid].top, 0) < MAXSTACKDEPTH * bid) {
        atomicAdd((int *)&stealStackStats[bid].top, MAXSTACKDEPTH);
    }
    SYNC_WARP;
    if(wtid == 0) {
        __threadfence();
#ifdef RACEY
        atomicExch_block(&stealStackStats[bid].locked, 0);
#else
        atomicExch(&stealStackStats[bid].locked, 0);
#endif
    }
    SYNC_WARP;
}

// See if any stacks have work left that we should wait for
__device__ bool workDone(StackStats *localStacks, StackStats *stealStacks)
{
    __shared__ char done[(NTHREADS + WARP_SIZE - 1) / WARP_SIZE];
    const int wtid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    if(wtid == 0) {
        done[wid] = 1;
        for(int i = 0; i < gridDim.x; ++i)
            if(atomicAdd((int*)&localStacks[i].workAvail, 0) > 0 || 
                atomicAdd((int*)&stealStacks[i].workAvail, 0) > 0) {
                done[wid] = 0;
                break;
            }
    }
    __syncthreads();
    return done[wid];
}

// Search stacks for ready nodes
__global__ void parTreeSearch(StackStats *localStackStats, Node *localStacks, 
    StackStats *stealStackStats, Node *stealStacks, Config config) 
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wtid = tid % WARP_SIZE;
    Node parent;
    parent.param[HEIGHT] = 0;
    int i = bid;
    /* tree search */
    do {
        parent.param[CHILDREN] = 0;
        // Have warp acquire lock for local stack
        SYNC_WARP;
        while(wtid == 0 && atomicCAS_block(&localStackStats[bid].locked, 0, 1) != 0);
        __threadfence_block();
        SYNC_WARP;

        if (atomicAdd((int*)&localStackStats[bid].workAvail, 0) > wtid) {

            *(unsigned int *)(parent.param) = atomicAdd_block((int *)&localStacks[
                (localStackStats[bid].top + wtid) % LOCAL_DEPTH + LOCAL_DEPTH * bid].param, 0);
            if(wtid == 0) {
                localStackStats[bid].top = (localStackStats[bid].top + min(WARP_SIZE, 
                    atomicAdd((int*)&localStackStats[bid].workAvail, 0))) % LOCAL_DEPTH + LOCAL_DEPTH * bid;
                int val = atomicSub((int*)&localStackStats[bid].workAvail, WARP_SIZE);
                if(val - WARP_SIZE < 0)
                    atomicExch((int*)&localStackStats[bid].workAvail, 0);
            }
        }
      
        SYNC_WARP;
        if(wtid == 0) {
            __threadfence_block();
            atomicExch_block(&localStackStats[bid].locked, 0);
        }
        SYNC_WARP;
      
        genChildren(&parent, localStackStats, localStacks, stealStackStats, stealStacks, config);
        // Check a steal stack now
        int blk = i;
        i = ((i + 1) % NBLOCKS);
        parent.param[CHILDREN] = 0;
        SYNC_WARP;
#ifdef RACEY
        while(wtid == 0 && atomicCAS_block(&stealStackStats[blk].locked, 0, 1) != 0);
#else
        while(wtid == 0 && atomicCAS(&stealStackStats[blk].locked, 0, 1) != 0);
#endif
        __threadfence();
        SYNC_WARP;
        if(atomicAdd((int*)&stealStackStats[blk].workAvail, 0) > wtid) {
            *(unsigned int *)(parent.param) = atomicAdd((int *)&stealStacks[
                (stealStackStats[blk].top + wtid) % MAXSTACKDEPTH + MAXSTACKDEPTH * blk].param, 0);
            if(wtid == 0) {
                stealStackStats[blk].top = (stealStackStats[blk].top + 
                    min(WARP_SIZE, atomicAdd((int*)&stealStackStats[blk].workAvail, 0)))
                    % MAXSTACKDEPTH + MAXSTACKDEPTH * blk;
              
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
        if(wtid == 0) {
            __threadfence();
#ifdef RACEY
            atomicExch_block(&stealStackStats[blk].locked, 0);
#else
            atomicExch(&stealStackStats[blk].locked, 0);
#endif
        }
        SYNC_WARP;
        genChildren(&parent, localStackStats, localStacks, stealStackStats, stealStacks, config);
    } while(!workDone(localStackStats, stealStackStats));
    /* tree search complete ! */
}
