/**
    Functions that enable the decomposition that are the same 
    for classic and swept versions.
*/

#include "decompCore.h"

void preSetDevice()
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
}

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



void parseArgs(json inJ, int argc, char *argv[]);
{
    // Will include div, tpb, type (classic vs swept), and file outputs.)
    // The numbers aren't right yet.

    if (argc>6)
    {
        for (int k=6; k<argc; k+=2)
        {
            inJ[argv[k]] = argv[k+1];
        }
    }
}
// // Topology
// int devcnt;
// int nthreads;

// // Geometry
// int tpb, tpbp, base;
// int dv, bk;
// int ht, htm, htp;
// int szState;

// // Iterator
// int tstep=1;

// // devicepointers
// states *dStateBase;
// states *dState[4];

void initArgs(json inJ);
{
    // Make Eq constants, states initial and hbound
    equationSpecificArgs(json inJ); 
    tpb = inJ["tpb"];
    tpb




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


    if (dimz.dt_dx > .21)
    {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.21 for stability." << endl;
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

