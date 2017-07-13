/**
    The Classic Functions for the stencil operation
*/

#include <cuda.h>
#include "Classic_Core.h"
#include "io/printer.h"
#include <mpi.h>

/** Classic kernel for simple decomposition of spatial domain.

    Uses dependent variable values in States_in to calculate States out.  If it's the predictor step, finalstep is false.  If it is the final step the result is added to the previous States_out value because this is RK2.

    @param States_in The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretizatio
    @param States_out The working array from the kernel call before last which either stores the predictor values or the full step values after the RHS is added into the solution.
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__
void
classicDecomp(states *state, int tstep)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, tstep)
}


//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, states *state)
{
    states *dState;

    cudaMalloc((void **)&dStates, sizeof(states)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dStates_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    cout << "Classic scheme" << endl;

    double t_eq = 0.0;
    int tstep = 0;
    double twrite = freq - QUARTER*dt;

    while (t_eq < t_end)
    {
        classicDecomp <<< bks,tpb >>> (dStates_in, tstep);
        //And swap!
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dStates_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dStates, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dStates);

    return t_eq;

}