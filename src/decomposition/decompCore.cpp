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

// void getDeviceInformation();
// {
//     cudaGetDeviceCount(nGpu);

//     if (nGpu)
//     {
//         cudaGetDeviceProp(&props);
//     }
    
//     nthreads = omp_get_num_procs();

//     // From this I want what GPUs each proc can see, and how many threads they can make
//     // This may require nvml to get the UUID of the GPUS, pass them all up to the 
//     // Master proc to decide which proc gets which gpu.
// }

/* 
    Takes any extra command line arguments which override json default args and inserts 
    them into the json type which will be read into variables in the next step.

    Arguments are key, value pairs all lowercase keys, no dash in front of arg.
*/
static bool mg = false;

void parseArgs(jsons inJ, int argc, char *argv[])
{
    if (argc>6)
    {
        for (int k=6; k<argc; k+=2)
        {
            inJ[argv[k]] = argv[k+1];
		// If it sets nW, flip the bool.
	    mg=true;
        }
    }
}

void initArgs(jsons inJ)
{
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.szState = sizeof(states);
    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.tpb = inJ["tpb"].asInt();
    cGlob.gpuA = inJ["gpuA"].asDouble();
    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.nX = inJ["nX"].asInt();

    // Derived quantities
    cGlob.xcpu = cGlob.nThreads * cGlob.tpb;
    cGlob.bks = (((double)cGlob.xcpu * cGlob.gpuA)/cGlob.tpb);
    cGlob.xg = cGlob.tpb * cGlob.bks;
    cGlob.gpuA = (double)cGlob.xg/(double)cGlob.tpb; // Adjusted gpuA.
    cGlob.xWave = (nprocs * cGlob.xcpu + cGlob.nGpu * cGlob.xg); 

    // Do it backward if you already know the waves. Else we get the waves from nX (which is just an approximation).
    if (mg)
    {
        cGlob.nWaves = inJ["nW"].asInt();
    }
    else
    {
        cGlob.nWaves = CEIL(cGlob.xWave, cGlob.nX);
    }

    cGlob.nX = cGlob.nWaves*cGlob.xWave;
    cGlob.tpbp = cGlob.tpb + 1;
    cGlob.base = cGlob.tpb + 2;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht - 1;
    cGlob.htp = cGlob.ht + 1;

    cGlob.dx = cGlob.lx/((double)cGlob.nX - 2.0); // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs(inJ);

    // Swept Always Passes!

    // If BCTYPE == "Dirichlet"
    if (!ranks[1]) cGlob.bCond[0] = false;
    if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.

}

void solutionOutput(states *outState, REAL tstamp, REAL xpt)
{
    for (int k=0; k<NVARS; k++)
    {
        //solution[outVars[k]][tstamp][xpt] = printout(k, outState);
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

