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

void classicPass(states *state)
{ 
    // Doesn't handle the CUDA case yet because tpb: make new array, four
    // pointers or indices based on number of points per node including GPU>
    // (pass 1->last element)
    if (!ranks[1]) MPI_Sendrecv(&state[1], szState, struct_type, ranks[0], TAGS(tstep), 
                                &state[tpbp], szState, struct_type, ranks[1], TAGS(tstep),
                                MPI_COMM_WORLD, &status); 
    // Pass second to last element -> first element.
    if (ranks[1] < lastproc) MPI_Sendrecv(&state[tpb], szState, struct_type, ranks[2],
                                TAGS(tstep), &state, szState, struct_type, ranks[0], TAGS(tstep),
                                MPI_COMM_WORLD, &status); 
}


//Classic Discretization wrapper.
double classicWrapper(double *xpts, states *state)
{
    cout << "Classic scheme" << endl;

    double t_eq = 0.0;
    int tstep = 1; //Starts at 1 (Initial condition is 0)
    double twrite = freq - QUARTER*dt;

    if (xgpu) // If there's no gpu assigned to the process this is 0.
    {
        int bks = xgpu/tpb;
        int gpui = xcpu/2;
        int gpuf = gpui + xgpu + 2; 
        states *state1 = &state[0];
        states *state2 = &state[gpuf];

        states *dState;

        cudaMalloc((void **)&dState, sizeof(states)*xgpu);

        // Copy the initial conditions to the device array.
        cudaMemcpy(dState, &state[gpui], sizeof(states)*(xgpu+2), cudaMemcpyHostToDevice);
        //They also need private sections of the array!

        classicStepCPU(state1, gpui);
        classicStepCPU(state2, gpui);

        while (t_eq < t_end)
        {
            classicDecomp <<< bks,tpb >>> (dState, tstep);

            //Swap GPU-CPU
            //Swap MPI

            classicPass(state);
            
            // Increment Counter
            t_eq += dt;
            tstep++

            if (t_eq > twrite)
            {
                cudaMemcpy(&state[gpui], dState, sizeof(states)*xgpu, cudaMemcpyDeviceToHost);
                for (int k in )
                solutionOutput(&state, t_eq, &xpts)

                twrite += freq;
            }
        }

        cudaMemcpy(&state[gpui], dState, sizeof(states)*xgpu, cudaMemcpyDeviceToHost);

        cudaFree(dState);
    }
    else
    {

        while (t_eq < t_end)
        {

            classicStepCPU(&state, gpui);
            t_eq += dt;
            tstep++
        }
    }

    return t_eq;

}