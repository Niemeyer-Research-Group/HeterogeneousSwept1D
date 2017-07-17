#ifndef EULERGLOBALS_H
#define EULERGLOBALS_H

#include <mpi.h>
#include <omp.h>
#include "json.hpp"

int ranks[3];
int nprocs;
int devcnt;
int nthreads;
int tstep;

struct geometry{
    int tpb;
    int ht[3];
    int base;
};

// Cuda Device Prop props;

json solution;
json timing;

void makeMPI(int argc, char* argv[]);

void topology();

void endMPI();

void eCheckIn (int dv, int tpb, int argc);

void initializeOutStreams();

void solutionOutput();

void timingOutput();

#endif