# DEF adds extra defines. Specificially -DNOS=1. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS=1"
DEF :=

NATLIB := -lm -std=c++11 -gencode arch=compute_35,code=sm_35 -L$(LD_LIBRARY_PATH) -restrict -O3 -w --ptxas-options=-v $(DEF)

DH := decomposition/*.h

CPREQ := solmain.cu utilities/jsoncpp.cpp

default: ./bin/heat ./bin/euler ./bin/cons

./bin/euler: $(CPREQ) equations/euler.h equations/heads.h $(DH)
	nvcc $(CPREQ) -o ./bin/euler -DEULER=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi

./bin/heat: $(CPREQ) equations/heat.h equations/heads.h $(DH)
	nvcc $(CPREQ) -o ./bin/heat -DHEAT=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi

./bin/cons: $(CPREQ) equations/constantF.h equations/heads.h $(DH)
	nvcc $(CPREQ) -o ./bin/cons -DCONS=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi