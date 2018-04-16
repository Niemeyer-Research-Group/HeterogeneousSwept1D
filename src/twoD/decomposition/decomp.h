/**
    DECOMP CORE And other things.
*/

#include <numeric>
#include "gpuDetector.h"
#include "json/jsons.h"
#include "mpi.h"

typedef Json::Value jsons;

#define TAGS(x) x & 32767

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

// MPI process properties
MPI_Datatype struct_type;
MPI_Request req[2];
MPI_Status stat[2];
int lastproc, nprocs, rank;

struct Location{
    const int x, y;
    Location(int xy, int xgrid, int ygrid): x(xy % xgrid), y(xy/ygrid)) {};
    Location(int x, int y): x(x), y(y) {};
};

struct Neighbor: public Location
{
    int owner;
    states *sender, *receiver;


}

// DO WE REALLY WANNA LOAD THIS UP AND THEN PASS IT TO THE KERNEL?
struct Region: public Location
{
    int owner;
    bool gpu;
    Location coordinate;
    jsons solution; 
    Neighbors n, s, e, w;
    states *state[]; // pts in region
    states **stateRows[]; // rows in region
    
    Region() {};
    Region(int x, int y, bool gpu) : gpu(gpu) Location(x, y) 
    {
        if (gpu){cudaHostAlloc();}
        else {malloc};
    };
    ~Region()
    void solutionOutput();
};

struct Globalism {
// Topology
    int nGpu, nPts, nWrite, hasGpu;
    double gpuA;

// Geometry
	int szState;
    int blockPts, xBlock, blockBase;
    int nRegion, xRegion, yRegion;
    int gridNodes, gridSidex, gridSidey;
    int blocksx, blocksy;
    int gpux, gpuy;
    int nx, ny;
    int ht, htm, htp;

// Iterator
    double tf, freq, dt, dx, dy, lx, ly;

} cGlob;

std::string fname = "GranularTime.csv";

jsons inJ;


void makeMPI(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
}

// I think this needs a try except for string inputs.
void parseArgs(int argc, char *argv[])
{
    if (argc>4)
    {
        std::string inarg;
        for (int k=4; k<argc; k+=2)
        {
            inarg = argv[k];
			inJ[inarg] = atof(argv[k+1]);
        }
    }
}

void setRegion(int affinity, int hasGpu, int xreg, int yreg, Region **regionals)
{
    int nRegions = 1 + hasGpu*(affinity - 1);
    regionals = malloc(nRegions * sizeof(Region)); 
    int regionMap[nProcs];
    int MPI_Allgather(&nRegions, 1, MPI_INT, &regionMap[0], 1, MPI_INT, MPI_COMM_WORLD);
    int tregion = 0;
    int gridlook[yreg][xreg];
    int k, i;

    for (k = 0; k<nProcs; k++)
    {
        for (i = 0; i<regionMap[k]; i++)
        {
            Location loca(treq, xreg, yreg); // Muy loco
            gridlook[loca.y][loca.x] = k;
            tregion++;
        }
    }

    int cnt = 0;
    for (k = 0; k<yreg; k++)
    {
        for (i = 0; i<xreg; i++)
        {
            if (gridlook[k][i] == rank)
            {
                regionals[cnt] = new *Region(k, i, OTHERSHIT);
            }
        }
    }
}

void initArgs()
{
    int *dimFinder;

    //VALUES THAT NEED NO INTROSPECTION

    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.ly = inJ["ly"].asDouble();
    cGlob.szState = sizeof(states);
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    // IF TPB IS NOT SQUARE AND DIVISIBLE BY 32 PICK CLOSEST VALUE THAT IS.
    int tpbReq = inJ["tpb"].asInt();
    cGlob.blockSide = std::sqrt(tpbReq);
    if (cGlob.blockSide % 8) cGlob.blockSide = 8 * (1 + (cGlob.blockSide/8)); 
    cGlob.blockPts = cGlob.blockSide * cGlob.blockSide;
    inJ["blockSide"] = cGlob.blockSide;
    cGlob.blockBase = cGlob.blockSide+2;
    inJ["blockBase"] = cGlob.blockBase;

    // FIND NUMBER OF GPUS AVAILABLE.
	cGlob.gpuA = inJ["gpuA"].asDouble(); // AFFINITY
	int ranker = rank;
	int sz = nprocs;

	if (!cGlob.gpuA)
    {
        cGlob.hasGpu = 0;
        cGlob.nGpu = 0;
    }
    else
    {
        cGlob.hasGpu = detector(ranker, sz, 0);
        MPI_Allreduce(&cGlob.hasGpu, &cGlob.nGpu, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        sz -= cGlob.nGpu;
    }

    cGlob.gridNodes = sz + (cGlob.nGpu * cGlob.gpuA);
    dimFinder = factor(cGlob.gridNodes);
    
    // WE JUST WANT THE DIMENSIONS TO BE RELATIVELY SIMILAR.
    while (dimFinder[1]/dimFinder[0] > 2)
    {
        if (!cGlob.nGpu)
        {
            std::cout << "You provided a prime number of processes and no GPU capability. That's just silly and you absolutely cannot do it because we must be able to form a rectangle from the number of processes on each device.  Retry!" << std::endl;
            exit();
        }
        cGlob.gpuA++;
        cGlob.gridNodes = sz + (cGlob.nGpu * cGlob.gpuA);
        dimFinder = factor(cGlob.gridNodes);
    }
    cGlob.gridSidex = dimFinder[0]; 
    cGlob.gridSidey = dimFinder[1];

    // Something that makes this work if you give it nRegion rather than nPts.
    cGlob.nPts = inJ["nPts"].asInt();
    cGlob.nRegion = cGlob.nPts/(cGlob.blockPts * cGlob.gridNodes);
    dimFinder = factor(cGlob.nRegion);

    while (dimFinder[1]/dimFinder[0] > 2)
    {
        cGlob.nRegion++;
        dimFinder = factor(cGlob.nRegion);
    }

    cGlob.nPts = cGlob.blockPts * cGlob.nRegion * cGlob.gridNodes;
    cGlob.xRegion = dimFinder[1];
    cGlob.yRegion = dimFinder[0];
    cGlob.blocksx = cGlob.xRegion * cGlob.gridSidex;
    cGlob.blocksy = cGlob.yRegion * cGlob.gridSidey;
    cGlob.nx = cGlob.blocksx * cGlob.blockSide;
    cGlob.ny = cGlob.blocksy * cGlob.blockSide;

    cGlob.ht = cGlob.blockSide/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;
    // Derived quantities

    cGlob.dx = cGlob.lx/(double)cGlob.nx; // Spatial step
    cGlob.dy = cGlob.ly/(double)cGlob.ny; // Spatial step
    cGlob.nWrite = cGlob.tf/cGlob.freq + 2;
    inJ["dx"] = cGlob.dx; // To send back to equation folder. 
    inJ["dy"] = cGlob.dy;

    HCONST.init(inJ);
    cudaMemcpyToSymbol(DCONST, &HCONST, sizeof(HCONST));

    if (!rank)  std::cout << rank <<  " - Initialized Arguments -" << std::endl;
}

// I THINK THIS SHOULD BE IN THE NODE STRUCT
void solutionOutput(states *outState, double tstamp, const int idx, const int idy)
{
    #ifdef NOS
        return; // Prevents write out in performance experiments so they don't take all day.
    #endif
    std::string tsts = std::to_string(tstamp);
    double xpt = indexer(cGlob.dx, idx, strt);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(outState + idx, k);
    }
}

struct mpiTime
{
    std::vector<double> times;
    double ti;
    std::string typ = "CPU";

    void tinit(){ ti = MPI_Wtime(); }

    void tfinal() { times.push_back((MPI_Wtime()-ti)*1000.0); }

    int avgt() { 
        return std::accumulate(times.begin(), times.end(), 0)/ (float)times.size();
        }
};

void atomicWrite(std::string st, std::vector<double> t)
{
    FILE *tTemp;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int k=0; k<nprocs; k++)
    {
        if (rank == k)
        {
            tTemp = fopen(fname.c_str(), "a+");
            fseek(tTemp, 0, SEEK_END);
            fprintf(tTemp, "\n%d,%s", rank, st.c_str());
            for (auto i = t.begin(); i != t.end(); ++i)
            {
                fprintf(tTemp, ",%4f", *i);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&struct_type);
	MPI_Finalize();
}
