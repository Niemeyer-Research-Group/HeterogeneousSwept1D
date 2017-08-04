/**
    Functions that enable the decomposition that are the same 
    for classic and swept versions.
*/

#include "decompCore.h"

//Always prepared for periodic boundary conditions.
void makeMPI(int argc, char* argv[])
{
    mpi_type(&struct_type);
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

void getDeviceInformation();
{
    cudaGetDeviceCount(nGpu);

    if (nGpu)
    {
        cudaGetDeviceProp(&props);
    }
    
    nthreads = omp_get_num_procs();

    // From this I want what GPUs each proc can see, and how many threads they can make
    // This may require nvml to get the UUID of the GPUS, pass them all up to the 
    // Master proc to decide which proc gets which gpu.
}

void delegateDomain()
{
    // Set shared memory banks to double if REAL is double.
    if (sizeof(REAL)>6 && xgpu) 
    {
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
}

/* 
    Takes any extra command line arguments which override json default args and inserts 
    them into the json type which will be read into variables in the next step.

    Arguments are key, value pairs all lowercase keys, no dash in front of arg.
*/
void parseArgs(json inJ, int argc, char *argv[]);
{
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

void initArgs(json inJ);
{
    try{
        equationSpecificArgs(json inJ); 
    }
    except
    {
        echeckIn(0, "Your input json is missing a key element.");
    }

    tpb = inJ["tpb"];
    gpuA = inJ["gpuA"];
    dt = inJ["dt"];
    tf = inJ["tf"];
    freq = inJ["freq"];
    nX = inJ["nX"];

    xg = ((tpb * gpuA)/32) * 32;  // Number of gpu spatial points.
    xcpu = nThreads * tpb;
    xWave = (nprocs * xcpu + nGpu * xg); // Number of points on a device x number of devices.
    nWaves = CEIL(xWave, nX);
    nX = nWaves*xWave; // Now it's an even wave of spatial points.
    
    enum
    {
        // If BCTYPE == "Dirichlet"
        if (!rank) bCond[0] = false;
        if (rank == lastproc) bCond[1] = false;
        // If BCTYPE == "Periodic"
            // Don't do anything.
    }


}
// THIS IS GREAT BUT YOU CAN'T PASS IT BACK BECAUSE TYPES!
void solutionOutput(REALthree outState, REAL tstamp, REAL xpt)
{
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tstamp][xpt] = printout(k, outVec); 
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

