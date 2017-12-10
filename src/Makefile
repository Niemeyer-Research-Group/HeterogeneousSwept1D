# DEF adds extra defines. Specificially -DNOS=1. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS=1"
DEF :=

include conf

MAKEPATH:=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

IPATH+=-I$(MAKEPATH)/equations -I$(MAKEPATH)/decomposition -I$(MAKEPATH)/utilities

# LPATH+=

BINPATH:=$(MAKEPATH)/bin
EQPATH:=$(MAKEPATH)/equations
DECOMPPATH:=$(MAKEPATH)/decomposition
UTILPATH:=$(MAKEPATH)/utilities

DH := $(DECOMPPATH)/*.h $(EQPATH)/heads.h

CPREQ := solmain.cu $(UTILPATH)/jsoncpp.cpp $(DH)

EQHEAD=$(BINPATH)/$(basename $(notdir $($(EQPATH)))

# OBJS=$(filter-out $(DH), $(EQHEAD))

default: $(OBJS)

$(OBJS): $(BINPATH)/%: $(EQPATH)%.h $(CPREQ) equations/heads.h
	nvcc $(CPREQ) -o $@ -include $< -DEULER=1 $(NATLIB) 

# ./bin/euler: : %.h $(CPREQ) equations/heads.h
# 	nvcc $(CPREQ) -o $@ -DEULER=1 $(NATLIB) 

# ./bin/heat: $(CPREQ) equations/heat.h equations/heads.h $(DH)
# 	nvcc $(CPREQ) -o ./bin/heat -DHEAT=1 $(NATLIB) 

# ./bin/cons: $(CPREQ) equations/constantF.h equations/heads.h
# 	nvcc $(CPREQ) -o ./bin/cons -DCONS=1 $(NATLIB)

### -------
# My Conf

# GPUARCH=35

# #Any additional include paths
# IPATH=-I/usr/include/mpi
# #Any additional library paths
# LPATH=

# CUDAFLAG=-gencode arch=compute_$(GPUARCH),code=sm_$(GPUARCH) -restrict --ptxas-options=-v
# CFLAGS =-O3 -std=c++11
# CLIBS=-lm -lmpi  