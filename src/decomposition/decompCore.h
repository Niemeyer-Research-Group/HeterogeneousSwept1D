#ifndef DECOMPCORE_H
#define DECOMPCORE_H

#include <mpi.h>
#include <omp.h>
#include "json.hpp"
#include "dummyheader.h"

#define TAGS(x) x & 32767

// MPI process properties
MPI_Status status;
MPI_Datatype struct_type;
int ranks[3];
int nprocs;
int lastproc;

// Topology
int nthreads, xgpu, xcpu;

// Geometry
int tpb, tpbp, base;
int ht, htm, htp;
int szState;

// Iterator
double t_end, freq, dt;
int tstep=1;

// Cuda Device Prop props;

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

void solutionOutput(REALthree outVec, REAL tstamp, REAL xpt);

void timingOutput(REAL timer, FILE *timeOut);

void endMPI();

#endif