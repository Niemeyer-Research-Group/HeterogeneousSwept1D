/**
	The Global Variables.  MPI admin variables, Solution structs, GPU pointers to constant and global memory.
*/

#ifndef GLOBALS_H
#define GLOBALS_H

#include <mpi.h>
#include <stdio.h>

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
