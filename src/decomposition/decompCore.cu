/**
    Functions that enable the decomposition that are the same 
    for classic and swept versions.
*/

#include "decompCore.h"

void topology(int *devcnt)
{
    cudaGetDeviceCount(devcnt);
    if (devcnt)
    {
        
    }
    //Check the node characteristics like whether it has a GPU
}

void initMPI(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	Lrank = myrank - 1; if(Lrank <     0) Lrank = size-1;
	Rrank = myrank + 1; if(Rrank == size) Rrank = 0;
}

void finMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

