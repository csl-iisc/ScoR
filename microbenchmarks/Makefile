NVCC	    := nvcc

NVCC_FLAGS  := -arch=compute_60
MB_DIR      := ./src
MB_EXEC_DIR := ./bin
MB_FILES    := $(wildcard $(MB_DIR)/*.cu)
MB_EXEC     := $(patsubst $(MB_DIR)/%.cu,$(MB_EXEC_DIR)/%,$(MB_FILES))

all: $(MB_EXEC)

$(MB_EXEC_DIR)/%: $(MB_DIR)/%.cu
	mkdir -p $(MB_EXEC_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	
clean:
	rm -f $(MB_EXEC_DIR)/*
