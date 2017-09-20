nvcc solmain.cu jsoncpp.cpp -o ./bin/heat -DEULER=1 -gencode arch=compute_35,code=sm_35 -O3 -restrict -std=c++11 -I/usr/include/mpi -lmpi -Xcompiler -fopenmp -lm -w --ptxas-options=-v
echo "EULER"
nvcc solmain.cu jsoncpp.cpp -o ./bin/heat -DHEAT=1 -gencode arch=compute_35,code=sm_35 -O3 -restrict -std=c++11 -I/usr/include/mpi -lmpi -Xcompiler -fopenmp -lm -w --ptxas-options=-v
echo "HEAT"
