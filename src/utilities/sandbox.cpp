/*
    Just a testing main program for the functions
*/

// Compile IT:  nvcc -o sandbx sandbox.cu -lm -restrict -gencode arch=compute_35,code=sm_35

//nvcc -o sandbx sandbox.cu ../equations/Euler/Euler_Device.cu -lm -rdc=true -gencode arch=compute_35,code=sm_35 
// gcc -I/usr/include/mpi -lmpi -o sndbx sandbox.cu -O3 -lm -gencode arch=compute_35,code=sm_35 --std=c++11

#include "json.hpp"
#include <iostream>
#include <fstream>

using json = nlohmann::json;

int main(int argc, char *argv[])
{
    std::ifstream imzep("test.json", std::ifstream::in);
    json myJ;
    imzep >> myJ;
    std::cout << myJ.dump(4) << std::endl;

    imzep.close();
    return 0;

    // int rank, devcnt;
    // cout << argc << endl;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // cudaGetDeviceCount(&devcnt);
    // cout << rank << " " << devcnt << endl;
    // MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Finalize();
}