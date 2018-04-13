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
    int x, y;
    Point(int x, int y): x(x), y(y) {}
};

struct Node : public Location
{
    Point dim;
    int n, s, e, w;
    Node(int x, int y) : Location(x, y) {}
};

struct Globalism {
// Topology
    int nGpu, nPts, nWrite, hasGpu;
    double gpuA;

// Geometry
	int szState;
    int blockPts, blockSide, blockBase;
    int nodeBlocks, nodeSidex, nodeSidey;
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
jsons solution;

//Always prepared for periodic boundary conditions.
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

void initArgs()
{
    int *dimFinder;

	cGlob.gpuA = inJ["gpuA"].asDouble();
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
        sz--;
    }

    int *tFactor;
    tFactor = factor(cGlob.gpuA);
    if (cGlob.gpuA > 2)
    {
        while (tFactor[0] == 1) // Means it's prime so it can't make a rectangle.
        {
            cGlob.gpuA++;
            tFactor = factor(cGlob.gpuA);
        }
    }
    cGlob.gpux = tFactor[0];
    cGlob.gpuy = tFactor[1];    
    // Wanted to standardize default values for these, but I realize that's not the point. Defaults are what the json is for.
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.ly = inJ["ly"].asDouble();
    cGlob.szState = sizeof(states);
    int tpbReq = inJ["tpb"].asInt();
    cGlob.blockSide = std::sqrt(tpbReq);
    cGlob.blockPts = cGlob.blockSide * cGlob.blockSide;
    inJ["blockSide"] = cGlob.blockSide;
    cGlob.blockBase = cGlob.blockSide+2;
    inJ["blockBase"] = cGlob.blockBase;

    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    cGlob.gridNodes = sz + (cGlob.nGpu * cGlob.gpuA);
    dimFinder = factor(cGlob.gridNodes);
    
    while (dimFinder[0] != 1)
    {
        if (!cGlob.nGpu)
        {
            std::cout << "You provided a prime number of processes and no GPU capability. That's just silly and you absolutely cannot do it.  Retry!" << std::endl;
        }
        cGlob.gpuA++;
        tFactor = factor(cGlob.gpuA);
        while (tFactor[0] == 1) // Means it's prime so it can't make a rectangle.
        {
            cGlob.gpuA++;
            tFactor = factor(cGlob.gpuA);
        }
        cGlob.gridNodes = sz + (cGlob.nGpu * cGlob.gpuA);
        dimFinder = factor(cGlob.gridNodes);
    }

    cGlob.gridSidex = dimFinder[0];
    cGlob.gridSidey = dimFinder[1];

    // Something that makes this work if you give it nodeBlocks rather than nPts.
    cGlob.nPts = inJ["nPts"].asInt();
    cGlob.nodeBlocks = cGlob.nPts/(cGlob.blockPts * cGlob.gridNodes);
    dimFinder = factor(cGlob.nodeBlocks);

    while (cGlob.dimFinder[0] != 1)
    {
        cGlob.nodeBlocks++;
        dimFinder = factor(cGlob.nodeBlocks);
    }
    cGlob.nPts = cGlob.blockPts * cGlob.nodeBlocks * cGlob.gridNodes;
    cGlob.nodeSidex = dimFinder[1];
    cGlob.nodeSidey = dimFinder[0];
    cGlob.blocksx = cGlob.nodeSidex * cGlob.gridSidex;
    cGlob.blocksy = cGlob.nodeSidey * cGlob.gridSidey;
    cGlob.nx = cGlob.blocksx * cGlob.blockSide;
    cGlob.ny = cGlob.blocksy * cGlob.blockSide;

    // -- Here ish

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
void solutionOutput(states *outState, double tstamp, const int idx, const int idxy)
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
