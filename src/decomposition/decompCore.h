#ifndef DECOMPCORE_H
#define DECOMPCORE_H

#include <mpi.h>
#include <omp.h>
#include <algorithm> //string utility
#include "json.hpp"
#include "dummyheader.h"

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

json solution;
json timing;

void makeMPI(int argc, char* argv[]);

void getDeviceInformation();

void delegateDomain(double *xpts, states *state);

// All additional options overwrite inJ default values.
void parseArgs(json inJ, int argc, char *argv[]);

// Now inJ gives values to variables.
void initArgs(json inJ);

void eCheckIn(int argc);

void solutionOutput(states *outVec, double tstamp, double xpt1);

void timingOutput(REAL timer, FILE *timeOut);

void endMPI();

#endif