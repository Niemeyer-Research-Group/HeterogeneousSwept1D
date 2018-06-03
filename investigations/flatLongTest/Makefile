

CUDAFLAGS :=  -gencode arch=compute_35,code=sm_35 -restrict -O3 -lm -std=c++11 -w --ptxas-options=-v

ifndef NOPLOT
	PLOTSS=-I/usr/include/python2.7 -lpython2.7
endif

ifdef PD
	EXTO:=PAD
else
	PD=
	EXTO:=
endif

# It will put the compiled programs in the bin subdirectory
PROJECT_DIR = $(shell pwd)
OUT_DIR := ./bin
#$(PROJECT_DIR)

default: $(OUT_DIR)/KSmain$(EXTO)

$(OUT_DIR)/%$(EXTO): %.cu $(PROJECT_DIR)/*.hpp
	nvcc $< -o $@ $(CUDAFLAGS) $(PLOTSS) $(PD)

clean:
	rm bin/*
