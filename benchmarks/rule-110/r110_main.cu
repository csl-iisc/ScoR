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
 * Implementation of Rule 110 Cellular Automaton
 * https://en.wikipedia.org/wiki/Rule_110
 ********************************************************************************************/

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#define MAIN
#include "r110_kernel.cuh"

/**************************************************
 *
 *                HOST FUNCTIONS
 *
 **************************************************/

void errorCheck(cudaError_t err, const char location[])
{
    if (err != cudaSuccess)
    {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "; at " << location << "\n";
        exit(1);
    }
}

/**************************************************
 *              INPUT DESCRIPTION
 * size = Number of elements in the automaton
 * steps = Number of iterations of rule 110
 *
 *************************************************/

void input(int &size, int &steps, DATA **arr)
{
    cin >> size >> steps;
    *arr = new DATA[size];

    for(int i = 0; i < size; ++i)
        cin >> (*arr)[i];
}

void outputValues(int size, DATA *arr)
{
    ofstream out("rule110-ans.txt");
    for(int i = 0; i < size; ++i)
        out << arr[i] << " ";
    out << "\n";
}

int main()
{
    int size, steps;
    // Declare host variables
    DATA *h_arr;
    // Input data
    input(size, steps, &h_arr);
    // Declare device variables
    DATA *d_arr, *d_copy;
    int *d_comp;

    errorCheck(cudaMalloc(&d_arr, sizeof(DATA) * size), "Malloc device arr");
    errorCheck(cudaMalloc(&d_copy, sizeof(DATA) * size), "Malloc device copy");
    errorCheck(cudaMalloc(&d_comp, sizeof(int) * NBLOCKS * NTHREADS), "Malloc device complete");
    errorCheck(cudaMemcpy(d_arr, h_arr, sizeof(DATA) * size, cudaMemcpyHostToDevice), "Memcpy device arr");
    errorCheck(cudaMemset(d_comp, 0, sizeof(int) * NBLOCKS * NTHREADS), "Memset complete");
    for(int i = 0; i < steps; ++i)
        rule110Kernel<<<NBLOCKS, NTHREADS>>>(d_arr, d_copy, d_comp, size, i + 1);
    errorCheck(cudaMemcpy(h_arr, d_arr, sizeof(DATA) * size, cudaMemcpyDeviceToHost), "Memcpy host arr");
    outputValues(size, h_arr);
    return 0;
}
