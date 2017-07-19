/**
    The Classic Functions for the stencil operation
*/

#include "classicCore.h"

/** Classic kernel for simple decomposition of spatial domain.

    Uses dependent variable values in States_in to calculate States out.  If it's the predictor step, finalstep is false.  If it is the final step the result is added to the previous States_out value because this is RK2.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, ts)
}

void classicStepCPU(states *state)
{
    for (int k=1; k<tpb+1; k++)
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
    if (!rank[1]) MPI_Sendrecv(&state[1], szState, struct_type, ranks[0], 1, 
                                &state[tpbp], szState, struct_type, ranks[1], 1,
                                MPI_COMM_WORLD, &status); 
    // Pass second to last element -> first element.
    if (rank[1] < lastproc) MPI_Sendrecv(&state[tpb], szState, struct_type, ranks[2], 1, 
                                &state, szState, struct_type, ranks[0], 1,
                                MPI_COMM_WORLD, &status); 
}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, states *state)
{
    states *dState;

    cudaMalloc((void **)&dStates, sizeof(states)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dState, state, sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    cout << "Classic scheme" << endl;

    double t_eq = 0.0;
    int tstep = 1; //Starts at 1 (Initial condition is 0)
    double twrite = freq - QUARTER*dt;

    //They also need private sections of the array!

    classicStepCPU(state);

    while (t_eq < t_end)
    {
        classicDecomp <<< bks,tpb >>> (dState, tstep);
        //And swap!
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dState, sizeof(states)*dv, cudaMemcpyDeviceToHost);

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dState, sizeof(states)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dState);

    return t_eq;

}