DIRS ?= 1dconv graph-coloring graph-connectivity matrix-multiplication reduction rule-110 uts

RACEY ?= 
ARCH ?= -arch=compute_70

all:
	mkdir -p ./bin/
	for file in $(DIRS); do \
		$(MAKE) -C $$file RACEY=$(RACEY) ARCH=$(ARCH); \
		mv $$file/$$file ./bin/$$file; \
	done
	
clean:
	for file in $(DIRS); do \
		$(MAKE) -C $$file clean; \
	done
	rm -r ./bin
