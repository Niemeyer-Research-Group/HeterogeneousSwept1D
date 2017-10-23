
NATLIB := -lm -std=c++11 -gencode arch=compute_35,code=sm_35 -L$(LD_LIBRARY_PATH) -restrict -O3

default: ./bin/heat ./bin/euler ./bin/cons

./bin/euler: solmain.cu utilities/jsoncpp.cpp equations/euler.h equations/heads.h decomposition/decomp.h decomposition/classic.h decomposition/swept.h
	nvcc solmain.cu utilities/jsoncpp.cpp -o ./bin/euler -DEULER=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi

./bin/heat: solmain.cu utilities/jsoncpp.cpp equations/heat.h equations/heads.h decomposition/decomp.h decomposition/classic.h decomposition/swept.h
	nvcc solmain.cu utilities/jsoncpp.cpp -o ./bin/heat -DHEAT=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi

./bin/cons: solmain.cu utilities/jsoncpp.cpp equations/constantF.h equations/heads.h decomposition/decomp.h decomposition/classic.h decomposition/swept.h
	nvcc solmain.cu utilities/jsoncpp.cpp -o ./bin/cons -DCONS=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi