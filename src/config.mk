#ESSENTIAL PATHS.  Sure most are probably on your system, but let's make it official

PREFIX          =$(dir $(shell pwd))
NVCC            =$(shell which nvcc)
MPICXX          =$(shell which mpiexec)
SOURCEPATH      =$(PREFIX)src
CC_ICUDAPATH    =
CC_IMPIPATH     =
CC_LMPIPATH     =
CC_LCUDAPATH    =
CUDAFLAGS       =-gencode arch=compute_35,code=sm_35 -restrict  --ptxas-options=-v
CFLAGS          =-O3 --std=c++11 -w
LIBMPI          =-lmpi 
LIBCUDA         =-lcudart -lcuda

# Additional PATHS
IPATH           =
LPATH           =

# Additional Libs
LIBS            =-lm 
