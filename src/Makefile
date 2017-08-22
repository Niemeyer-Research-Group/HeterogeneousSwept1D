# Simple makefile for all test problems for 1D Swept Rule
# -maxrregcount=73 

MPICC=$(PREP) mpicc
MPILD=$(PREP) mpic++
NVCC=$(PREP) nvcc

# It will put the compiled programs in the bin subdirectory within the directory
# containing the makefile
MPIPATH := /usr/include/mpi
PROJECTDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
OUTDIR := $(PROJECTDIR)bin
EQDIR := $(PROJECTDIR)equations
COREDIR := $(PROJECTDIR)decomposition
CONFS := $(PROJECTDIR)eqconf.conf
EQS := $(file < $(CONFS))
EQDIRS := $(addprefix $(EQDIR)/, $(EQS))

# MPICFLAGS=-I${MPI_HOME}/include
# CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include
# EQDIRS := ${sort ${dir ${wildcard ${EQDIR}/*/}}}
# EQHEAD = $(EQDIRS)*.h
# APPOBJ := $(OUTDIR)/${lastword ${subst /, ,$(EQDIRS)}}

# IFNVCC = -lmpi -Xcompiler -fopenmp -Xptxas=-v -restrict -march=native
# GENCODE := -lcudart -lcuda

PARALLELLIB := -Xcompiler -fopenmp -lmpi -gencode arch=compute_35,code=sm_35
NATLIB := -lm -O3 -std=c++11 
INC := -I$(PROJECTDIR)utilities -I$(PROJECTDIR)equations -I$(MPIPATH)
CUDAFLAGS := $(NATLIB) $(INC) $(PARALLELLIB) 
COREDEP := $(PROJECTDIR)solver.cu $(wildcard $(COREDIR)/*.cpp)

.PHONY: $(EQS)

$(EQS): %: $(EQDIRS)/%.cu $(COREDEP) 
	$(NVCC) $^ -o $(OUTDIR)/$@ $(CUDAFLAGS) -D=$@

# all: $(EQS)

# define build_rule
# $1: $(COREDEP) 
# 	$(MPICC) $(COREDEP) $(EQDIR)/$1/$1.cu -o $(OUTDIR)/$1 -D=$1 $(CUDAFLAGS)
# endef

# $(foreach eq,$(EQS), $(eval $(call build_rule,$(eq))))

# .PHONY: printvar 
