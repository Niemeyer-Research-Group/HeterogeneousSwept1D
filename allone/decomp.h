/*
---------------------------
    DECOMP CORE
---------------------------
*/

#include <string>

#define TAGS(x) x & 32767

#define CEIL(x, y)  (x + y - 1) / y 

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

// MPI process properties
MPI_Datatype struct_type;
MPI_Request req[2];
int lastproc, nprocs, ranks[3];

struct globalism {
// Topology
    int nThreads, nWaves, nGpu, nX;  
    int xg, xcpu, xWave;
    bool hasGpu;
    double gpuA;

// Geometry
    int tpb, tpbp, base, bks;
    int ht, htm, htp;
    int szState;

// Iterator
    double tf, freq, dt, dx, lx;
    bool bCond[2] = {true, true}; // Initialize passing both sides.
};

globalism cGlob;

jsons inJ;
jsons solution;
jsons timing;

//Always prepared for periodic boundary conditions.
void makeMPI(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
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

// I think this needs a try except for string inputs.
void parseArgs(int argc, char *argv[])
{
    std::cout << argc << std::endl;
    if (argc>5)
    {
        for (int k=5; k<argc; k+=2)
        {
            inJ[argv[k]] = atof(argv[k+1]);   
        }
    }
}

void initArgs()
{
    using namespace std;
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
    cGlob.xcpu = cGlob.nThreads * cGlob.tpb;    // How many spatial points can fit on one wave on a CPU.
    cGlob.bks = (((double)cGlob.xcpu * cGlob.gpuA)/cGlob.tpb); // How many blocks should be launched on a GPU.
    cGlob.xg = cGlob.tpb * cGlob.bks;
    cGlob.gpuA = (double)cGlob.xg/(double)cGlob.xcpu; // Adjusted gpuA.
    cGlob.xWave = (nprocs * cGlob.xcpu + cGlob.nGpu * cGlob.xg); 

    // Do it backward if you already know the waves. Else we get the waves from nX (which is just an approximation).
    cout << "nX " << cGlob.nX << endl;
    cout << "nprocs: " << nprocs << endl;
    cout << "gpuA " << cGlob.gpuA << endl;
    cout << "nGPU: " << cGlob.nGpu << endl;
    cout << "----------"  << endl;

    if (inJ["nW"].asInt() == 0)
    {
        cGlob.nWaves = CEIL(cGlob.nX, cGlob.xWave);
        cout << "No nWaves wasn't set previous" << endl;
    }
    else
    {
        cout << "Yes nWaves was set previous " << inJ["nW"].asInt() << endl;
        cGlob.nWaves = inJ["nW"].asInt();
    }

    cGlob.nX = cGlob.nWaves*cGlob.xWave;
    cGlob.tpbp = cGlob.tpb + 1;
    cGlob.base = cGlob.tpb + 2;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht - 1;
    cGlob.htp = cGlob.ht + 1;

    cGlob.dx = cGlob.lx/((double)cGlob.nX - 2.0); // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    
    cout << cGlob.dx << endl;
    double mydx = inJ["dx"].asDouble();
    cGlob.xg *= cGlob.nWaves;
    cGlob.xcpu *= cGlob.nWaves;

    cout << "After:" << endl;
    
    cout << "nX " << cGlob.nX << endl;
    cout << "xg + xc " << cGlob.xg + cGlob.xcpu << endl;
    cout << "nWaves " << cGlob.nWaves << endl;
    cout << "xWave " << cGlob.xWave << endl;
    cout << "bks " << cGlob.bks << endl;
    cout << "gpuA " << cGlob.gpuA << endl;

    cin >> cGlob.freq;

    equationSpecificArgs(inJ);

    // Swept Always Passes!

    // If BCTYPE == "Dirichlet"
    if (!ranks[1]) cGlob.bCond[0] = false;
    if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.
    cout << inJ << endl;

}

void solutionOutput(states *outState, REAL tstamp, REAL xpt)
{
    std::string tsts = std::to_string(tstamp);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(k, outState);
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}