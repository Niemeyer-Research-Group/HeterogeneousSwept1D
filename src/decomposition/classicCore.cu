/**
    The Classic Functions for the stencil operation
*/

#include "classicCore.h"

/** 
    Classic kernel for simple decomposition of spatial domain.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, ts)
}

void classicStepCPU(states *state, int numx)
{
    #pragma omp parallel for
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep)
    }
}


void classicPassLeft(states *state, int idxend)
{
    if (bCond[0])
    {
        MPI_Bsend(&state[1], 1, struct_type, ranks[0], TAGS(tstep),
                MPI_COMM_WORLD);

        MPI_recv(&state[0], 1, struct_type, ranks[0], TAGS(tstep*2), 
                MPI_COMM_WORLD, &status);
    }
                                     
}

void classicPassRight(states *state, int idxend)
{
    if (bCond[1]) 
    {
        MPI_Bsend(&state[idxend-1], 1, struct_type, ranks[2], TAGS(tstep*2),
                MPI_COMM_WORLD);

        MPI_recv(&state[idxend], 1, struct_type, ranks[2], TAGS(tstep), 
                MPI_COMM_WORLD, &status);
    }
}

//Classic Discretization wrapper.
double classicWrapper(states *state, double *xpts)
{
    cout << "Classic Decomposition" << endl;
    int *bpt, bsize = 50*szState;
    MPI_Buffer_attach(malloc(bsize), bsize);

    tstep = 1; 
    double t_eq = 0.0;
    double twrite = freq - QUARTER*dt;

    if (xg) // If there's no gpu assigned to the process this is 0.
    {
        int xc = xcpu/2, xcp = xc+1, xcpp = xc+2;
        int xgp = xg+1, xgpp = xg+2;
        int gpusize =  szState * xgpp;
        int cpuzise = szState * xcpp;

        states *state1, *state2;
        cudaHostAlloc((void **)&state1, cpusize);
        cudaHostAlloc((void **)&state2, cpusize);
        memcpy(state1, state, cpusize)
        memcpy(state2, state + xg + xc, cpusize)

        // Four streams for four transfers to and from cpu.
        cudaStream_t st1, st2, st3, st4;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);
        cudaStreamCreate(&st4);

        states *dState, *hState;

        cudaHostAlloc((void **)&hState, gpusize);
        cudaMalloc((void **)&dState, gpusize;

        // Copy the initial conditions to the device array.
        cudaMemcpy(dState, state + xc, gpusize, cudaMemcpyHostToDevice);

        while (t_eq < t_end)
        {
            classicStepCPU(state1, xcp);
            classicStepCPU(state2, xcp);
            classicDecomp <<< bks,tpb >>> (dState, tstep);

            // Host to device first.
            cudaMemcpyAsync(dState, state1 + xc, szState, cudaMemcpyHostToDevice, st1);
            cudaMemcpyAsync(dState + xgp, state2 + 1, szState, cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(state1 + xcp, dState + 1, szState, cudaMemcpyDeviceToHost, st3);
            cudaMemcpyAsync(state2, dState + xg, szState, cudaMemcpyDeviceToHost, st4);

            classicPassRight(state2, xcp);
            classicPassLeft(state1, xcp);
            
            // Increment Counter and timestep
            if (MODULA(tstep)) t_eq += dt;
            tstep++

            if (t_eq > twrite)
            {
                cudaMemcpy(&hState, dState, gpubytes, cudaMemcpyDeviceToHost);

                #pragma omp parallel for
                for (int k=1; k<xcp; k++) solutionOutput(state1[k]->Q[0], xpts[k], t_eq);
                
                #pragma omp parallel for
                for (int k=1; k<xcp; k++) solutionOutput(state2[k]->Q[0], xpts[k+xc+xg], t_eq);

                #pragma omp parallel for
                for (int k=1; k<xgp; k++) solutionOutput(hState[k]->Q[0], xpts[k+xc], t_eq);

                twrite += freq;
            }
        }

        cudaMemcpy(&hState, dState, gpubytes, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);

        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state1[k]->Q[0], xpts[k], t_eq);
        
        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state2[k]->Q[0], xpts[k+xc+xg], t_eq);

        #pragma omp parallel for
        for (int k=1; k<xgp; k++) solutionOutput(hState[k]->Q[0], xpts[k+xc], t_eq);

        cudaFree(dState);
        cudaFreeHost(hState);
        cudaFreeHost(state1);
        cudaFreeHost(state2);
    }
    else
    {
        while (t_eq < t_end)
        {
            int xcp = xcpu + 1;

            classicStepCPU(state, xcp);
            if (MODULA(tstep)) t_eq += dt;
            tstep++

            classicPassRight(state, xcp);
            classicPassLeft(state, xcp);)

            if (t_eq > twrite)
            {
                #pragma omp parallel for
                for (int k=1; k<xcp; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);
                twrite += freq;
            }
            
        }
        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);
    }
    return t_eq;
}

