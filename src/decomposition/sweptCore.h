// IF IT DOESN'T COMPILE TEMPLATE ALL THE KERNELS.

#ifndef SWEPTCORE_H
#define SWEPTCORE_H

#include "decompCore.h"

__global__ void upTriangle(states *state, int tstep);

__global__ void downTriangle(states *state, int tstep);

__global__ void wholeDiamond(states *state, int tstep);

void upTriangleCPU(states *state);

void downTriangleCPU(states *state);

void wholeDiamondCPU(states *state);

void passSwept(states *state, int idxend, int dr);

double sweptWrapper(states *state, double *xpts, int *tstep);

#endif 