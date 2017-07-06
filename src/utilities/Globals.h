/**
	The Global Variables.  MPI admin variables, Solution structs, GPU pointers to constant and global memory.
*/

#ifndef GLOBALS_H
#define GLOBALS_H

#include <mpi.h>
#include <cuda.h>
#include "Euler_Device.cuh"
#include "../../../cuda/myVectorTypes.h"

/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

typedef REAL double;
typedef REALthree double3;

struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int base; // Length of node + stencils at end (4)
    int idxend; // Last index (number of spatial points - 1)
};

struct states{
    REALthree Q; // Full Step state variables
    REAL Pr; // First step Pressure ratio
	REALthree Qmid; // Midpoint state variables
	REAL Prmid;
};

/*
	============================================================
	MPI ADMIN VARIABLES
	============================================================
*/

int myrank;
int Lrank;
int Rrank;
int size;
int debug;

void initMPI(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	Lrank = myrank - 1; if(Lrank <     0) Lrank = size-1;
	Rrank = myrank + 1; if(Rrank == size) Rrank = 0;
}

void finMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

#endif
