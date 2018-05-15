#ESSENTIAL PATHS.  Sure most are probably on your system, but let's make it official

PREFIX          =$(MSCRATCH)/HeterogeneousSwept1D
NVCC            =$(CUDAPATH)/bin/nvcc
MPICXX          =$(MPIPATH)/bin/mpicxx)
SOURCEPATH      =$(PREFIX)/src
CC_ICUDAPATH    =$(CUDAPATH)/include
CC_IMPIPATH     =$(MPIPATH)/include
CC_LMPIPATH     =
CC_LCUDAPATH    =
CUDAFLAGS       =-gencode arch=compute_35,code=sm_35 -restrict  --ptxas-options=-v
CFLAGS          =-O3 --std=c++11 -w
LIBMPI          =-lmpich -lopa -lmpl -lrt -lpthread 
LIBCUDA         =-lcudart -lcuda

# Additional PATHS
IPATH           =
LPATH           =$(MPIPATH)/lib $(INTELLIB) $(CUDAPATH)/lib64

# Additional Libs
LIBS            =-lm 
