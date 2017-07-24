/*
    Just a testing main program for the functions
*/

// Compile IT:  nvcc -o sandbx sandbox.cu -lm -restrict -gencode arch=compute_35,code=sm_35

//nvcc -o sandbx sandbox.cu ../equations/Euler/Euler_Device.cu -lm -gencode arch=compute_35,code=sm_35 
// gcc -o sndbx sandbox.cpp -O3 -lm --std=c++11
// Yeah it works with g++ though (5.4)
// You also need a dummy variable when you use the value.

#include <iostream>
#include <fstream>
#include <mpi.h>
#include <omp.h>
#include "json.hpp"

using jsons = nlohmann::json;


// "MV2_COMM_WORLD_LOCAL_RANK"
#define ENV_LOCAL_RANK		"OMPI_COMM_WORLD_LOCAL_RANK"

void SetDeviceBeforeInit()
{
	char * localRankStr = NULL;
	int rank = 0, devCount = 0;

	// We extract the local rank initialization using an environment variable
	if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
	{
		rank = atoi(localRankStr);		
	}

	cudaGetDeviceCount(&devCount);
    int mdev = rank % devCount;
	cudaSetDevice(mdev);
    cout << "Before Init: Process " << rank << " has device " << mdev << endl;
}

// Test json
// int main(int argc, char *argv[])
// {
//     std::ifstream imzep("test.json", std::ifstream::in);
//     jsons myJ;
//     imzep >> myJ;
//     std::cout << myJ.dump(4) << std::endl;

//     int dt = myJ["dt"]; 
//     imzep.close();

//     std::cout << dt*5 << std::endl;
//     return 0;
// }


// Test device sight.
int mainint argc, char *argv[])
{
    int rank, mydev;
    cout << argc << endl;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaGetDevice(&mydev);
    cout << "After Init: Process " << rank << " has device " << mydev << endl;
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
