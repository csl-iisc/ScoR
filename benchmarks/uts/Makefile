RACEY    ?= 
NBLOCKS  ?= 60
NTHREADS ?= 256

NVCC ?= nvcc
ARCH ?= -arch=compute_60
INCL := -I. 

ifeq ($(RACEY),1)
  NVCC_FLAGS  := $(ARCH) -std=c++11 -D RACEY
else
  NVCC_FLAGS  := $(ARCH) -std=c++11
endif

FILES := uts_main.cu uts_kernel.cu
EXEC  := uts

all: $(EXEC)

$(EXEC): $(FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCL) -D NTHREADS=$(NTHREADS) -D NBLOCKS=$(NBLOCKS) $^ -o $@
	
clean:
	rm -f $(EXEC)
