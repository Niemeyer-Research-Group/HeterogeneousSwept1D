/**
    The Classic Functions for the stencil operation
*/

// Perhaps http://www.cplusplus.com/reference/unordered_map/unordered_map/insert/
// For json.

#include "classicCore.h"

/** 
    Classic kernel for simple decomposition of spatial domain.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, ts);
}

void classicStepCPU(states *state, int numx, int tstep)
{
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}

void classicPassLeft(states *state, int idxend, int tstep)
{   
    if (cGlob.bCond[0])
    {
        MPI_Isend(&state[1], 1, struct_type, ranks[0], TAGS(tstep),
                MPI_COMM_WORLD, &req[0]);

        MPI_Recv(&state[0], 1, struct_type, ranks[0], TAGS(tstep+100), 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
                                     
}

void classicPassRight(states *state, int idxend, int tstep)
{
    if (cGlob.bCond[1]) 
    {
        MPI_Isend(&state[idxend-1], 1, struct_type, ranks[2], TAGS(tstep+100),
                MPI_COMM_WORLD, &req[1]);

        MPI_Recv(&state[idxend], 1, struct_type, ranks[2], TAGS(tstep), 
                MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    }
}

// We are working with the assumption that the parallelism is too fine to see any benefit.
// Still struggling with the idea of the local vs parameter arrays.
// Classic Discretization wrapper.
double classicWrapper(states **state, double **xpts, int *tstep)
{
    std::cout << "Classic Decomposition" << std::endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        const int xc = cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;
        const int cpuzise = cGlob.szState * xcpp;

        states *dState;
        
        cudaMalloc((void **)&dState, gpusize);
        // Copy the initial conditions to the device array.
        // This is ok, the whole array has been malloced.
        cudaMemcpy(dState, state[1], gpusize, cudaMemcpyHostToDevice);

        // Four streams for four transfers to and from cpu.
        cudaStream_t st1, st2, st3, st4;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);
        cudaStreamCreate(&st4);

        while (t_eq < cGlob.tf)
        {
            classicStep <<< cGlob.bks, cGlob.tpb >>> (dState, *tstep);

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicStepCPU(state[0], xcp, *tstep);
                }
                #pragma omp section
                {
                    classicStepCPU(state[2], xcp, *tstep);
                }
            }
            
            // Host to device first.
            # pragma omp parallel sections num_threads(3)
            {
                #pragma omp section
                {
                    cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
                    cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
                    cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
                    cudaMemcpyAsync(state[0], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4);
                }
                #pragma omp section
                {
                    classicPassRight(state[2], xcp);
                }
                #pragma omp section
                {
                    classicPassLeft(state[0], xcp);
                }
            }
            
            // Increment Counter and timestep
            if (MODULA(*tstep)) t_eq += cGlob.dt;
            tstep++;

            if (t_eq > twrite)
            {
                cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);                
                for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
                for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

                twrite += cGlob.freq;
            }
        }

        cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);
        
        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
        for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
        for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

        cudaFree(dState);
    }
    else
    {
        int xcp = cGlob.xcpu + 1;

        while (t_eq < cGlob.tf)
        {

            classicStepCPU(state[0], xcp, *tstep);
            if (MODULA(*tstep)) t_eq += cGlob.dt;
            *tstep++;

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicPassRight(state[0], xcp);
                }
                #pragma omp section
                {
                    classicPassLeft(state[0], xcp);
                }
            }

            if (t_eq > twrite)
            
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
                twrite += cGlob.freq;
            }

        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
    }
    return t_eq;
}
