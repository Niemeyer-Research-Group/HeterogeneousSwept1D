#ESSENTIAL PATHS.  Sure most are probably on your system, but let's make it official

PREFIX          =$(HOME)/Documents/1_SweptRuleResearch/hSweep
NVCC            =$(shell which nvcc)
MPICXX          =$(shell which mpicxx)
SOURCEPATH      =$(PREFIX)/src
CC_ICUDAPATH    =/usr/local/cuda/include
CC_IMPIPATH     =/usr/local/openmpi/include
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
