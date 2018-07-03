/*
---------------------------
    DECOMP CORE
---------------------------
*/

#include <numeric>
#include <mpi.h>
#include "gpuDetector.h"
#include "json/json.h"

#define TAGS(x) x & 32767

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

// MPI process properties
MPI_Datatype struct_type;
MPI_Request req[2];
MPI_Status stat[2];
int lastproc, nprocs, ranks[3];

struct Globalism {
    // Topology
    int nGpu, nX;
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

    // Iterator
    double tf, freq, dt, dx, lx;
    bool bCond[2] = {true, true};
};

std::string fname = "GranularTime.csv";

Globalism cGlob;
jsons inJ;
jsons solution;

// Always prepared for periodic boundary conditions.
void makeMPI(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks[1]);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
    ranks[0] = ((ranks[1])>0) ? (ranks[1]-1) : (nprocs-1);
    ranks[2] = (ranks[1]+1) % nprocs;
}

// I think this needs a try except for string inputs.
void parseArgs(int argc, char *argv[])
{
    if (argc>4)
    {
        std::string inarg;
        for (int k=4; k<argc; k+=2)
        {
			// if (!ranks[1]) std::cout << "CL Arg " << ranks[1] << " " << argv[k] << " " << argv[k+1] << std::endl;
            inarg = argv[k];
			inJ[inarg] = atof(argv[k+1]);
        }
    }
}

// gpuA = gBks/cBks

void initArgs()
{
	cGlob.gpuA = inJ["gpuA"].asDouble();
	int ranker = ranks[1];
	int sz = nprocs;

	if (!cGlob.gpuA)
    {
        cGlob.hasGpu = 0;
        cGlob.nGpu = 0;
    }
    else
    {
        cGlob.hasGpu = detector(ranker, sz, 1);
        MPI_Allreduce(&cGlob.hasGpu, &cGlob.nGpu, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    cGlob.lx = inJ["lx"].asDouble();
    cGlob.szState = sizeof(states);
    cGlob.tpb = inJ["tpb"].asInt();

    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();

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
        cGlob.cBks = std::round(cGlob.nX/(cGlob.tpb*(nprocs + cGlob.nGpu * cGlob.gpuA)));
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
    if (ranks[1] == 0) cGlob.bCond[0] = false;
    if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.
    if (!ranks[1])  std::cout << "Initialized Arguments" << std::endl << "-------------" << std::endl;

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

void endMPI()
{
    MPI_Type_free(&struct_type);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
