# Scoped Racey (ScoR) Benchmark Suite
ScoR is a benchmark suite comprised of GPU programs that make use of scoped synchronization. The goal is to demonstrate how scoped synchronization can be used in CUDA, and showcase different types of scoped races that can arise. Consequently, code provided may not be the most optimal implementation.    
Contains:
* 32 microbenchmarks
* 7 benchmarks   

For further details refer to our paper:   
- A. K. Kamath, A. A. George, A. Basu. **ScoRD: A Scoped Race Detector for GPUs.** In _Proceedings of 47th IEEE/ACM International Symposium on Computer Architecture (ISCA), 2020._ [[Paper]](https://www.csa.iisc.ac.in/~arkapravab/papers/isca20_ScoRD.pdf) [[Video]](https://www.csa.iisc.ac.in/~arkapravab/papers/ScoRD_talk.mp4) [[Bibtex]](https://www.computer.org/csdl/api/v1/citation/bibtex/proceedings/1lsaqzyS9u8/09138958)   

## Compilation details
This code was compiled using CUDA 8.0 with the compute_60 flag set, and tested using GPGPU-Sim.   
Makefiles have been included in each subfolder, which should be edited to fit your requirements. 

## Copyright
Copyright (c) 2020 Indian Institute of Science   
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
- Neither the names of Computer Systems Lab, Indian Institute of Science, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

NOTE: The file reduction.cu is derived from the CUDA Sample Code. 
As such, this file is bound by the corresponding legal terms and conditions set forth separately (contained in NVIDIA_EULA.txt in benchmarks/reduction folder).
