#ifndef CLASSICCORE_H
#define CLASSICCORE_H

#include "decompCore.h"

__global__ void classicStep(states *state, int tstep);

void classicStepCPU(states *state, int tstep, int tpb);

void classicPass(states *state, int tpb, int rank, bool dr);

classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, states *state);

#endif