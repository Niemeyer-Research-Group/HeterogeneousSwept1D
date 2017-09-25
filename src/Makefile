NATLIB := -lm -std=c++11 -gencode arch=compute_35,code=sm_35 -w
# -restrict -O3

default: ./bin/heat ./bin/euler

./bin/euler: solmain.cu jsoncpp.cpp euler.h heads.h decomp.h classic.h swept.h
	nvcc solmain.cu jsoncpp.cpp -o ./bin/euler -DEULER=1 $(NATLIB) -I/usr/include/mpi -lmpi -Xcompiler -fopenmp 

./bin/heat: solmain.cu jsoncpp.cpp heat.h heads.h decomp.h classic.h swept.h
	nvcc solmain.cu jsoncpp.cpp -o ./bin/heat -DHEAT=1 $(NATLIB) -I/usr/include/mpi -lmpi -Xcompiler -fopenmp
