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
    int nGpu, nX;  
    int xg, xcpu;
    bool hasGpu;
    double gpuA;

// Geometry
    int tpb, tpbp, base;
    int cBks, gBks;
    int ht, htm, htp;
    int szState;

// Iterator
    double tf, freq, dt, dx, lx;
    bool bCond[2] = {true, true}; 
    // Initialize passing both sides.
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


// I think this needs a try except for string inputs.
void parseArgs(int argc, char *argv[])
{
    std::cout << argc << std::endl;
    if (argc>4)
    {
        for (int k=4; k<argc; k+=2)
        {
            inJ[argv[k]] = atof(argv[k+1]);   
        }
    }
}

// gpuA = gBks/cBks

void initArgs()
{
    using namespace std;
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.szState = sizeof(states);
    cGlob.tpb = inJ["tpb"].asInt();
    
    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.gpuA = inJ["gpuA"].asDouble();
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    if (inJ["nX"].asInt() == 0)
    {
        if (inJ["cBks"].asInt() == 0)
        {
            cGlob.gBks = inJ['gBks'].asInt();
            cGlob.cBks = cGlob.gBks/cGlob.gpuA;
        }
        else 
        {
            cGlob.cBks = inJ['cBks'].asInt();
            cGlob.gBks = cGlob.cBks*cGlob.gpuA;
        }
        if (cGlob.cBks<2) cGlob.cBks = 2; // Floor for cpu blocks per proc.
        if (cGlob.cBks & 1) cGlob.cBks++;
        cGlob.nX = cGlob.tpb * (nprocs * cGlob.cBks + cGlob.nGpu * cGlob.gBks);
    }
    else
    {
        cGlob.nX = inJ["nX"].asInt();
        cGlob.cBks = cGlob.nX/(cGlob.tpb*(nprocs + cGlob.nGpu * cGlob.gpuA));
        if (cGlob.cBks & 1) cGlob.cBks++;
        cGlob.gBks = cGlob.gpuA*cGlob.cBks;
    }
    
    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;
    // Derived quantities
    cGlob.xcpu = cGlob.cBks * cGlob.tpb;  
    cGlob.xg = cGlob.gBks * cGlob.tpb;  

    inJ["gpuAA"] = (double)cGlob.cBks/(double)cGlob.gBks; // Adjusted gpuA.

        // Different schemes!
    cGlob.dx = cGlob.lx/((double)cGlob.nX - 2.0); // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs();

    // Swept Always Passes!

    // If BCTYPE == "Dirichlet"
    if (!ranks[1]) cGlob.bCond[0] = false;
    if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.
    cout << inJ << endl;

}

void solutionOutput(states *outState, double tstamp, int idx, int strt)
{
    std::string tsts = std::to_string(tstamp);
    double xpt = indexer(cGlob.dx, idx, strt);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(outState, k);
    }
}



void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}