/*
---------------------------
    DECOMP CORE
---------------------------
*/

#include <numeric>
#include "gpuDetector.h"

#define TAGS(x) x & 32767

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

// MPI process properties
MPI_Datatype struct_type;
MPI_Request req[2];
MPI_Status stat[2];
int lastproc, nprocs, ranks[3];

struct globalism {
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

globalism cGlob;
jsons inJ;
jsons solution;

//Always prepared for periodic boundary conditions.
void makeMPI(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks[1]);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
	cGlob.tpb = 128;
    ranks[0] = ((ranks[1])>0) ? (ranks[1]-1) : (nprocs-1);
    ranks[2] = (ranks[1]+1) % nprocs;
}

// NO PARSING ARGS.  HARDWIRE!