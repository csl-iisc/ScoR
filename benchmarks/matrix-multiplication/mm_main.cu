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
 * Implementation of Matrix Multiplication using Threadfence Locks
 *
 * Each block contains a subpart of a row of the input matrix.
 *
 ********************************************************************************************/

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#include "mm_kernel.cuh"

/**************************************************
 *
 *                HOST FUNCTIONS
 *
 **************************************************/

void errorCheck(cudaError_t err, const char location[])
{
    if (err != cudaSuccess) {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "; at " << location << "\n";
        exit(1);
    }
}

/**************************************************
 *              INPUT DESCRIPTION
 *
 * rA = # rows in matrix A
 * cA = # columns in matrix A = # rows in matrix B
 * cB = # columns in matrix B
 *
 * Input matrix should be in row-major format
 *************************************************/

void input(int &rA, int &cA, int &cB, datatype **matA, datatype **matB, datatype **matC)
{
    cin >> rA >> cA >> cB;
    *matA = new datatype[rA * cA];
    *matB = new datatype[cA * cB];
    *matC = new datatype[rA * cB];

    for(int i = 0; i < rA * cA; ++i)
        cin >> (*matA)[i];
    for(int i = 0; i < cA * cB; ++i)
        cin >> (*matB)[i];
    for(int i = 0; i < rA * cB; ++i)
        (*matC)[i] = 0;
}

void outputValues(int rows, int cols, datatype *matC)
{
    ofstream out("matrix-ans.txt");
    for(int i = 0; i < rows * cols; ++i) {
        out << matC[i] << " ";
        if(i % cols == cols - 1)
            out << "\n";
    }
}

int main()
{
    int rA, cA, cB;
    // Declare host variables
    datatype *h_matA, *h_matB, *h_matC;
    // Input data
    input(rA, cA, cB, &h_matA, &h_matB, &h_matC);
    // Declare device variables
    datatype *d_matA, *d_matB, *d_matC, *d_sh_lock, *d_tempC, *d_g_lock;

    errorCheck(cudaMalloc(&d_matA, sizeof(datatype) * rA * cA), "Malloc device matA");
    errorCheck(cudaMalloc(&d_matB, sizeof(datatype) * cA * cB), "Malloc device matB");
    errorCheck(cudaMalloc(&d_matC, sizeof(datatype) * rA * cB), "Malloc device matC");
    errorCheck(cudaMalloc(&d_sh_lock, sizeof(int) * NBLOCKS * WARP_SIZE), "Malloc device sh_lock");
    errorCheck(cudaMalloc(&d_g_lock, sizeof(int) * 1024), "Malloc device global lock");
    errorCheck(cudaMalloc(&d_tempC, sizeof(datatype) * NBLOCKS * WARP_SIZE), "Malloc device tempC");

    
    errorCheck(cudaMemset(d_g_lock, 0, sizeof(int) * 1024), "Memset g_lock");
    errorCheck(cudaMemcpy(d_matA, h_matA, sizeof(datatype) * rA * cA, cudaMemcpyHostToDevice), "Memcpy device matA");
    errorCheck(cudaMemcpy(d_matB, h_matB, sizeof(datatype) * cA * cB, cudaMemcpyHostToDevice), "Memcpy device matB");
    errorCheck(cudaMemcpy(d_matC, h_matC, sizeof(datatype) * rA * cB, cudaMemcpyHostToDevice), "Memcpy device matC");

    matMultKernel<<<NBLOCKS, NTHREADS>>>(d_matA, d_matB, d_matC, rA, cA, cB, d_sh_lock, d_tempC, d_g_lock);

    errorCheck(cudaMemcpy(h_matC, d_matC, sizeof(datatype) * rA * cB, cudaMemcpyDeviceToHost), "Memcpy host matC");
    outputValues(rA, cB, h_matC);
    return 0;
}
