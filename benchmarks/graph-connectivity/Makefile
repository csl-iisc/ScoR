RACEY    ?= 
NBLOCKS  ?= 15
NTHREADS ?= 400

NVCC ?= nvcc
ARCH ?= -arch=compute_60
INCL := -I. 

ifeq ($(RACEY),1)
  NVCC_FLAGS  := $(ARCH) -std=c++11 -D RACEY
else
  NVCC_FLAGS  := $(ARCH) -std=c++11
endif

FILES := gcon_main.cu gcon_kernel.cu
EXEC  := graph-connectivity

all: $(EXEC)

$(EXEC): $(FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCL) -D NTHREADS=$(NTHREADS) -D NBLOCKS=$(NBLOCKS) $^ -o $@
	
clean:
	rm -f $(EXEC)
