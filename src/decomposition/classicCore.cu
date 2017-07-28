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
    tstep++;
}

void classicPassLeft(states *state, int idxend)
{
    MPI_Sendrecv(&state[1], szState, struct_type, ranks[0], TAGS(tstep), 
                                    &state[idxend-1], szState, struct_type, ranks[1], TAGS(tstep),
                                    MPI_COMM_WORLD, &status); 
}

void classicPassRight(states *state, int idxend)
{
    MPI_Sendrecv(&state[idxend], szState, struct_type, ranks[2],
                                TAGS(tstep), &state, szState, struct_type, ranks[0], TAGS(tstep),
                                MPI_COMM_WORLD, &status); 
}

void classicPass(states *state, int idxend)
{
    if (!ranks[1]) classicPassLeft(states *state, int idxend); 

    if (ranks[1] < lastproc) classicPassRight(states *state, int idxend);
}

//Classic Discretization wrapper.
double classicWrapper(states *state, double *xpts)
{
    cout << "Classic Decomposition" << endl;

    double t_eq = 0.0;
    int tstep = 1; //Starts at 1 (Initial condition is 0)
    double twrite = freq - QUARTER*dt;

    if (xgpu) // If there's no gpu assigned to the process this is 0.
    {
        int bks = xgpu/tpb;
        int gpui = xcpu/2;
        int gpuf = gpui + xgpu + 2; 
        states *state1 = &state;
        states *state2 = &state[gpuf];
        int gpubytes = sizeof(states)*(xgpu + 2));

        states *dState, *hState;

        cudaHostAlloc((void **)&dState, gpubytes);

        cudaMalloc((void **)&dState, gpubytes);

        // Copy the initial conditions to the device array.
        cudaMemcpy(dState, &state[gpui], gpubytes, cudaMemcpyHostToDevice);
        //They also need private sections of the array!

        classicStepCPU(state1, gpui+1);
        classicStepCPU(state2, gpui+1);

        while (t_eq < t_end)
        {
            classicDecomp <<< bks,tpb >>> (dState, tstep);

            classicPassRight(state2, gpui+1);
            classicPassLeft(state1, gpui+1);
            
            // Increment Counter and timestep
            if (MODULA(tstep)) t_eq += dt;
            tstep++

            if (t_eq > twrite)
            {
                cudaMemcpy(&hState, dState, sizeof(states)*xgpu, cudaMemcpyDeviceToHost);

                #pragma omp parallel for
                for (int k=1; k<gpui; k++) solutionOutput(state1[k]->Q[0], xpts[k], t_eq);
                
                #pragma omp parallel for
                for (int k=1; k<gpui; k++) solutionOutput(state2[k]->Q[0], xpts[k+gpuf], t_eq);

                #pragma omp parallel for
                for (int k=1; k<xgpu; k++) solutionOutput(hState[k]->Q[0], xpts[k+gpui], t_eq);

                twrite += freq;
            }
        }

        cudaMemcpy(&hState[gpui], dState, sizeof(states)*xgpu, cudaMemcpyDeviceToHost);

        #pragma omp parallel for
        for (int k=1; k<gpui; k++) solutionOutput(state1[k]->Q[0], xpts[k], t_eq);
        
        #pragma omp parallel for
        for (int k=1; k<gpui; k++) solutionOutput(state2[k]->Q[0], xpts[k+gpuf], t_eq);

        #pragma omp parallel for
        for (int k=1; k<xgpu; k++) solutionOutput(hState[k]->Q[0], xpts[k+gpui], t_eq);

        cudaFree(dState);
        cudaFreeHost(hState);
    }
    else
    {

        while (t_eq < t_end)
        {

            classicStepCPU(state, xcpu + 1);
            if (MODULA(tstep)) t_eq += dt;
            tstep++
            classicPass(state, xcpu + 1);

            if (t_eq > twrite)
            {

                #pragma omp parallel for
                for (int k=1; k<xcpu+1; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);

                twrite += freq;
            }
            
        }
        #pragma omp parallel for
        for (int k=1; k<xcpu+1; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);
    }
    return t_eq;
}

