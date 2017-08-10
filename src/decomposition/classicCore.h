#ifndef CLASSICCORE_H
#define CLASSICCORE_H

#include "decompCore.h"

__global__ void classicStep(states *state, int tstep);

void classicStepCPU(states *state, int numx);

void classicPassLeft(states *state, int idxend);

void classicPassRight(states *state, int idxend);

void classicPass(states *state, int idxend);

double classicWrapper(states *state, double *xpts, int *tstep);

#endif