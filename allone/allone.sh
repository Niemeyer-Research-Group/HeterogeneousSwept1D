nvcc solmain.cu jsoncpp.cpp -o ./bin/euler -gencode arch=compute_35,code=sm_35 -std=c++11 -I/usr/include/mpi -lmpi -Xcompiler -fopenmp -lm -w
