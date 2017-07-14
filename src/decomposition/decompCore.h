/**

*/
#ifndef EULERGLOBALS_H
#define EULERGLOBALS_H

#include "mpi.h"

int rank;
int size;

void topology(int *devcnt);

void initMPI(int argc, char* argv[]);

void finMPI();

#endif