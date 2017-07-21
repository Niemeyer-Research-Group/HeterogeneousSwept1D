#ifndef DECOMPCORE_H
#define DECOMPCORE_H

#include <mpi.h>
#include <omp.h>
#include "json.hpp"
#include "dummyheader.h"

// YES THESE SHOULD BE GLOBAL.  I STAND BY IT!
using json = nlohmann::json;

// MPI process properties
MPI_Status status;
MPI_Datatype struct_type;
int ranks[3];
int nprocs;
int lastproc;

// Topology
int devcnt;
int nthreads;

// Geometry
int tpb, tpbp, base;
int dv, bk;
int ht, htm, htp;
int szState;

// Iterator
int tstep=1;

// devicepointers
states *dStateBase;
states *dState[4];

// Cuda Device Prop props;

json solution;
json timing;

void makeMPI(int argc, char* argv[]);

void topology();

void eCheckIn(int argc);

void solutionOutput(REALthree outVec, REAL tstamp, REAL xpt);

void timingOutput(REAL timer, FILE *timeOut);

void endMPI();

#endif