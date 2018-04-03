
#include "wave.h"

// Each process gets its own passing structure.  Should contain buffers to send and receive and 
struct Classic{

    Classic(){};
}

struct Passer{
    double *north, *south, *east, *west; // Buffers for pass
    int nt, st, et, wt; // ids for procs to pass TO
    int nf, sf, ef, wf; // ids to receive FROM

    Passer(int pid, int sz){
        // Malloc and get ids.
    };

    ~Passer(){};
}

__global__ void classicStep(states *state, const int ts)
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x + 1; 
    int gidy = blockDim.y * blockIdx.y + threadIdx.y + 1; 
    eq.stepUpdate(state, gidx, gidy, ts);
}

void classicStepCPU(states *state, const int numx, const int tstep)
{
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}