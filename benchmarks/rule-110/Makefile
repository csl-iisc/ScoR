RACEY    ?= 
NBLOCKS  ?= 15
NTHREADS ?= 1024

NVCC ?= nvcc
ARCH ?= -arch=compute_60
INCL := -I. 

ifeq ($(RACEY),1)
  NVCC_FLAGS  := $(ARCH) -std=c++11 -D RACEY
else
  NVCC_FLAGS  := $(ARCH) -std=c++11
endif

FILES := r110_main.cu r110_kernel.cu
EXEC  := rule-110

all: $(EXEC)

$(EXEC): $(FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCL) -D NTHREADS=$(NTHREADS) -D NBLOCKS=$(NBLOCKS) $^ -o $@
	
clean:
	rm -f $(EXEC)
