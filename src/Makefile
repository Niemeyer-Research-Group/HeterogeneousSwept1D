NATLIB := -lm -std=c++11 -gencode arch=compute_35,code=sm_35 -w
# -restrict -O3

default: ./bin/heat ./bin/euler

./bin/euler: solmain.cu utilities/jsoncpp.cpp equations/euler.h equations/heads.h decomposition/decomp.h decomposition/classic.h decomposition/swept.h
	nvcc solmain.cu utilities/jsoncpp.cpp -o ./bin/euler -DEULER=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi -Xcompiler -fopenmp 

./bin/heat: solmain.cu utilities/jsoncpp.cpp equations/euler.h equations/heads.h decomposition/decomp.h decomposition/classic.h decomposition/swept.h
	nvcc solmain.cu utilities/jsoncpp.cpp -o ./bin/heat -DHEAT=1 $(NATLIB) -I/usr/include/mpi -I./equations -I./decomposition -I./utilities -lmpi -Xcompiler -fopenmp
