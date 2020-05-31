# Scoped Racey Benchmark Suite 
The goal is to demonstrate how scoped synchronization can be used in CUDA, and also showcase different scoped races that can arise. Consequently, code provided may not be the most optimal implementation.    
Contains:
* 32 microbenchmarks
* 7 applications   

For further details refer to our paper:   
_Aditya K Kamath, Alvin A. George, Arkaprava Basu. "ScoRD: A Scoped Race Detector for GPUs" In the proceedings of 47th IEEE/ACM International Symposium on Computer Architecture, 2020._   

## Compilation details
This code was compiled using CUDA 8.0 with the compute_60 flag set, and tested using GPGPU-Sim.   
Makefiles have been included in each subfolder, which should be edited to fit your requirements. 

## Copyright
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


NOTE: The file reduction.cu is derived from the CUDA Sample Code. 
As such, this file is bound by the corresponding legal terms and conditions
set forth separately (contained in NVIDIA_EULA.txt in the benchmarks folder).
