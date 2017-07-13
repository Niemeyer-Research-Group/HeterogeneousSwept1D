#ifndef CLASSICCORE_H
#define CLASSICCORE_H

__global__ void classicStep(states *state, int tstep);

classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, states *state);

#endif