/**
    Functions that enable the decomposition that are the same 
    for classic and swept versions.
*/

#include "decompCore.h"

//Always prepared for periodic boundary conditions.
void makeMPI(int argc, char* argv[])
{
    mpi_type(&struct_type);
    // read_json();  Perhaps?
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks[1]);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
    szState = sizeof(states);
    base = tpb+2;
    tpbp = tpb+1;
    ht = tpb/2;
    htm = ht-1;
	ranks[0] = (ranks[1]) ? ranks[1]-1 : lastproc; 
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

void eCheckIn (int argc)
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

// THIS IS GREAT BUT YOU CAN'T PASS IT BACK BECAUSE TYPES!
void solutionOutput(REALthree outState, REAL tstamp, REAL* xpt)
{
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tstamp][xpt] = printout(k, outVec); 
    }
}

// THIS IS OK BECAUSE WE'RE NOT GOING TO DO IT IN AN MPI PROCESS.
void timingOutput(REAL timer, FILE *timeOut)
{
    //READ json first.
    timing[dv] = timer;
}


void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

