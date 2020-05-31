NVCC	    := nvcc
RACEY       := 
NBLOCKS     := 
NTHREADS    := 

ifeq ($(RACEY),1)
  NVCC_FLAGS  := -arch=compute_60 -std=c++11 -D RACEY
else
  NVCC_FLAGS  := -arch=compute_60 -std=c++11
endif

APP_DIR      := ./src
APP_EXEC_DIR := ./bin
APP_FILES    := $(wildcard $(APP_DIR)/*.cu)
APP_EXEC     := $(patsubst $(APP_DIR)/%.cu,$(APP_EXEC_DIR)/%,$(APP_FILES))

all: $(APP_EXEC)

$(APP_EXEC_DIR)/%: $(APP_DIR)/%.cu
	mkdir -p $(APP_EXEC_DIR)
	$(NVCC) $(NVCC_FLAGS) -D NTHREADS=$(NTHREADS) -D NBLOCKS=$(NBLOCKS) -o $@ $<
	
clean:
	rm -f $(APP_EXEC_DIR)/*
