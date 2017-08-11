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
	ranks[0] = (ranks[1]-1) % nprocs;
    ranks[2] = (ranks[1]+1) % nprocs;
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

void initArgs(json inJ);
{
    cGlob.lx = inJ["lx"]
    cGlob.szState = sizeof(states);
    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.tpb = inJ["tpb"];
    cGlob.gpuA = inJ["gpuA"];
    cGlob.dt = inJ["dt"];
    cGlob.tf = inJ["tf"];
    cGlob.freq = inJ["freq"];
    cGlob.nX = inJ["nX"];

    // Do it backward if you already know the waves.
    if (inJ.count("nWaves"))
    {

    }

    // This part needs rework.
    cGlob.xg = ((cGlob.tpb * cGlob.gpuA)/32) * 32;  // Number of gpu spatial points.
    cGlob.gpuA = cGlob.xg/cGlob.tpb; 
    cGlob.xcpu = cGlob.nThreads * cGlob.tpb;
    cGlob.xWave = (nprocs * cGlob.xcpu + cGlob.nGpu * cGlob.xg); // Number of points on a device x number of devices.
    cGlob.nWaves = CEIL(cGlob.xWave, cGlob.nX);
    cGlob.nX = cGlob.nWaves*cGlob.xWave; // Now it's an even wave of spatial points.

    cGlob.tpbp = cGlob.tpb + 1;
    cGlob.base = cGlob.tpb + 2;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht - 1;
    cGlob.htp = cGlob.ht + 1;
    cGlob.bks = cGlob.xg/cGlob.tpb;

    cGlob.dx = cGlob.lx/((double)cGlob.nX - 2.0); // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs(json inJ); 

    // Maybe here assign gpus
    // return bool
    // hasGpu = gpuAssign();



    // // Swept Always Passes!
    // enum
    // {
    //     // If BCTYPE == "Dirichlet"
    //     if (!ranks[1]) cGlob.bCond[0] = false;
    //     if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    //     // If BCTYPE == "Periodic"
    //         // Don't do anything.
    // }
}

void solutionOutput(states *outState, REAL tstamp, REAL xpt)
{
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tstamp][xpt] = printout(k, outState); 
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

