# Simple makefile for all test problems for 1D Swept Rule
# -maxrregcount=73

MPICC=$(PREP) mpicc
MPILD=$(PREP) mpic++
NVCC=$(PREP) nvcc

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
PROJECTDIR := $(shell pwd)
OUTDIR := $(PROJECTDIR)/bin
EQDIR := $(PROJECTDIR)/equations

# MPICFLAGS=-I${MPI_HOME}/include
# CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

GENCODE := -gencode arch=compute_35,code=sm_35
PARALLELLIB := -lmpi
NATLIB := -lm -restrict -O3 -dc -rdc=true
CROSSLIB := -Xcompiler --ptxas-options=-v
INC = -I$(PROJECTDIR)/utilities -I/usr/include/mpi

CUDAFLAGS = $(GENCODE) $(NATLIB) $(INC) $(PARALLELLIB) 

default:	$(OUTDIR)/euler.o

$(OUTDIR)/euler.o:	$(EQDIR)/Euler/EulerDevice.cu
	$(NVCC) -o $@ $< $(CUDAFLAGS)

# CUDAFLAGS = -gencode arch=compute_35,code=sm_35 -lm -restrict -O3 -Xcompiler -fopenmp --ptxas-options=-v
# CUDAFLAGSDOUBLE := $(CUDAFLAGS) -DREAL=double -DREALtwo=double2 -DREALthree=double3 -DREALfour=double4


# default:	$(OUT_DIR)/EulerSingleOut $(OUT_DIR)/EulerDoubleOut

# $(OUT_DIR)/EulerSingleOut:	Euler1D_SweptShared.cu
# 	nvcc -o $@ $< $(CUDAFLAGS)

# $(OUT_DIR)/EulerDoubleOut:	Euler1D_SweptShared.cu
# 	nvcc -o $@ $< $(CUDAFLAGSDOUBLE)
