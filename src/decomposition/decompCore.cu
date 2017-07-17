/**
    Functions that enable the decomposition that are the same 
    for classic and swept versions.
*/

#include "decompCore.h"

void makeMPI(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks[1]);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	ranks[0] = (ranks[1] > 0) ? nprocs-1 : ranks[1]-1; 
    ranks[2] = (ranks[1] == nprocs) ? 0 : ranks[1]+1; 
}

void topology()
{
    cudaGetDeviceCount(devcnt);

    if (devcnt)
    {
        cudaGetDeviceProp(&props);
    }

    nthreads = omp_get_num_procs();
}

eCheckIn void (int dv, int tpb, int argc)
{
    if (argc < 8)
	{
        std::cout << "NOT ENOUGH ARGUMENTS:" << std::endl;
		std::cout << "The Program takes 8 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << std::endl;
        std::cout << "Algorithm type, Variable Output File, Timing Output File (optional)" << std::endl;
		exit(-1);
	}

	if ((dv & (tpb-1) != 0) || (tpb&31) != 0)
    {
        std::cout << "INVALID NUMERIC INPUT!! "<< std::endl;
        std::cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << std::endl;
        exit(-1);
    }

}

void solutionOutput(char *outfile, REALthree outvector)
{

    fwr << "Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << std::endl;

    fwr << "Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << std::endl;

    fwr << "Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << std::endl;

    fwr << "Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << std::endl;

}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

