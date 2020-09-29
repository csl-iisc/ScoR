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
 * Implementation of Parallel Graph Connectivity Computation
 *
 * Based on:
 * M. Sutton, T. Ben-Nun and A. Barak
 * Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)
 *
 * Modified to enable "Work-Stealing" between blocks
 *
 ********************************************************************************************/

/**************************************************
 *
 *               KERNEL FUNCTIONS
 *
 **************************************************/

__global__ 
void initKernel(int *head, int *tail, int *graphComponents, int *bases, int *blockIds);

__global__ 
void linkKernel(int *head, int *tail, int *edgeSetU, int *edgeSetV, int *graphComponents, int *bases, int *blockIds);

__global__ 
void compressKernel(int *head, int *tail, int *graphComponents, int *bases, int *blockIds);
