#ifndef DECOMPCORE_H
#define DECOMPCORE_H

#include <mpi.h>
#include <omp.h>
#include <functional>
#include "json.hpp"
#include "dummyheader.h"

// YES THESE SHOULD BE GLOBAL 
// and state should not because state can't be allocated on the stack.

// MPI process properties
MPI_Status status;
MPI_Datatype struct_type;
int ranks[3];
int nprocs;
int lastproc;
int ppnC, ppnG;

// Topology
int devcnt;
int nthreads;
double gpuAffinity;

// Geometry
int tpb, tpbp, base;
int dv, bk;
int ht, htm, htp;
int szState;

// Iterator
int tstep;

// Cuda Device Prop props;

// ppnC = dv/(nprocs + devcnt*gpuAffinity)
// ppnG = gpu*ppnC;  
// THEN ROUND ppnG TO THE NEAREST MULTIPLE OF 32 AND RECALCULATE.
// I

void makeMPI(int argc, char* argv[]);

void topology();

void eCheckIn(int argc);

void solutionOutput(REALthree outState, REAL tstamp, REAL* xpt);

void timingOutput(REAL timer, FILE *timeOut);

void endMPI();

#endif