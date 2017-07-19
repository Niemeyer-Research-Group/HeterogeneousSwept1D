#ifndef DECOMPCORE_H
#define DECOMPCORE_H

#include <mpi.h>
#include <omp.h>
#include "json.hpp"

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
int ht, htm, htp;
int szState;

// Iterator
int tstep;

// Cuda Device Prop props;

json solution;
json timing;

void makeMPI(int argc, char* argv[]);

void topology();

void eCheckIn(int dv, int argc);

void initializeOutStreams();

void solutionOutput();

void timingOutput();

void endMPI();

#endif