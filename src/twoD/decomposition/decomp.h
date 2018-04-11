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
    int nGpu, nPts;
    int xg, xcpu;
    int xStart;
    int nWrite;
    int hasGpu;
    double gpuA;

// Geometry
	int szState;
    int tpb, tpbp, base;
    int cBks, gBks;
    int ht, htm, htp;
    int *dimNode, *dimBlocks;

// Iterator
    double tf, freq, dt, dx, lx;
    bool bCond[2] = {false, false};
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


    // Wanted to standardize default values for these, but I realize that's not the point. Defaults are what the json is for.
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.ly = inJ["ly"].asDouble();
    cGlob.szState = sizeof(states);
    int tpbReq = inJ["tpb"].asInt();
    cGlob.blockSide = std::sqrt(tpbReq);
    cGlob.tpb = cGlob.blockSide * cGlob.blockSide;

    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.nNodes = sz + (cGlob.nGpu * cGlob.gpuA);
    cGlob.dimNode = factor(cGlob.nNodes);
    
    while (cGlob.dimNode[0] != 1)
    {
        if (!nGpu)
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
        cGlob.nNodes = sz + (cGlob.nGpu * cGlob.gpuA);
        cGlob.dimNode = factor(cGlob.nNodes);
    }

    cGlob.nPts = inJ["nPts"].asInt();
    cGlob.dimBlocks = factor(cGlob.cBlocks);

    // -- Here ish


    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    if (inJ["nX"].asInt() == 0)
    {
        if (inJ["cBks"].asInt() == 0)
        {
            cGlob.gBks = inJ["gBks"].asInt();
            cGlob.cBks = cGlob.gBks/cGlob.gpuA;
        }
        else
        {
            cGlob.cBks = inJ["cBks"].asInt();
            cGlob.gBks = cGlob.cBks*cGlob.gpuA;
        }
        if (cGlob.cBks<2) cGlob.cBks = 2; // Floor for cpu blocks per proc.
        if (cGlob.cBks & 1) cGlob.cBks++;
    }
    else
    {
        cGlob.nX = inJ["nX"].asInt();
        cGlob.cBks = round(cGlob.nX/(cGlob.tpb*(nprocs + cGlob.nGpu * cGlob.gpuA)));
        if (cGlob.cBks & 1) cGlob.cBks++;
        cGlob.gBks = cGlob.gpuA*cGlob.cBks;
    }
    // Need to reset this after figuring out partitions.

    cGlob.nX = cGlob.tpb * (nprocs * cGlob.cBks + cGlob.nGpu * cGlob.gBks);

    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;
    // Derived quantities
    cGlob.xcpu = cGlob.cBks * cGlob.tpb;
    cGlob.xg = cGlob.gBks * cGlob.tpb;

    // inJ["gpuAA"] = (double)cGlob.gBks/(double)cGlob.cBks; // Adjusted gpuA.
    inJ["cBks"] = cGlob.cBks;
    inJ["gBks"] = cGlob.gBks;
    inJ["nX"] = cGlob.nX;
    inJ["xGpu"] = cGlob.xg;
    inJ["xCpu"] = cGlob.xcpu;

    // Different schemes!
    cGlob.dx = cGlob.lx/(double)cGlob.nX; // Spatial step
    cGlob.nWrite = cGlob.tf/cGlob.freq + 2;
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs(inJ);

    // Swept Always Passes!

    // If BCTYPE == "Dirichlet"
    if (rank == 0) cGlob.bCond[0] = false;
    if (rank == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.
    if (!rank)  cout << rank <<  " - Initialized Arguments" << endl;

}

void solutionOutput(states *outState, double tstamp, int idx, int strt)
{
    std::string tsts = std::to_string(tstamp);
    double xpt = indexer(cGlob.dx, idx, strt);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(outState + idx, k);
    }
}

void writeOut(states **outState, double tstamp)
{
    #ifdef NOS
        return; // Prevents write out in performance experiments so they don't take all day.
    #endif
    static const int ax[2] = {cGlob.xcpu/2, cGlob.xg};
    static const int bx[3] = {cGlob.xStart, cGlob.xStart+ax[0], cGlob.xStart+ax[0]+ax[1]};
    int k;

    if (cGlob.hasGpu)
    {
        for (int i=0; i<3; i++)
        {
            for(k=1; k<=ax[i&1]; k++)
            {
                solutionOutput(outState[i], tstamp, k, bx[i]);
            }
        }
    }
    else
    {
        for(k=1; k<=cGlob.xcpu; k++)
        {
            solutionOutput(outState[0], tstamp, k, cGlob.xStart);
        }
    }
}


struct cudaTime
{
    std::vector<double> times;
    cudaEvent_t start, stop;
	float ti;
    std::string typ = "GPU";

    cudaTime() {
        cudaEventCreate( &start );
	    cudaEventCreate( &stop );
    }
    ~cudaTime()
    {
        cudaEventDestroy( start );
	    cudaEventDestroy( stop );
    }

    void tinit(){ cudaEventRecord( start, 0); }

    void tfinal(){ 
        cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&ti, start, stop);
        times.push_back(ti); 
        }

    int avgt() { 
        return std::accumulate(times.begin(), times.end(), 0)/ times.size();
        }
};

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
