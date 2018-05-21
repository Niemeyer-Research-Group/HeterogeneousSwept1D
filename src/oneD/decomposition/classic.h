/**
---------------------------
    CLASSIC CORE
---------------------------
*/

/**
    The Classic Functions for the stencil operation
*/

/**
    Classic kernel for simple decomposition of spatial domain.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/

#ifdef GTIME
#define TIMEIN  timeit.tinit()
#define TIMEOUT timeit.tfinal()
#define WATOM atomicWrite(timeit.typ, timeit.times)
#else
#define TIMEIN  
#define TIMEOUT 
#define WATOM
#endif

using namespace std;

__global__ void classicStep(states *state, const int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)
    stepUpdate(state, gid, ts);
}

void classicStepCPU(states *state, const int numx, const int tstep)
{
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}

void classicPass(states *putSt, states *getSt, int tstep)
{
    int t0 = TAGS(tstep), t1 = TAGS(tstep + 100);
    int rnk;

    MPI_Isend(putSt, 1, struct_type, ranks[0], t0, MPI_COMM_WORLD, &req[0]);

    MPI_Isend(putSt + 1, 1, struct_type, ranks[2], t1, MPI_COMM_WORLD, &req[1]);

    MPI_Recv(getSt + 1, 1, struct_type, ranks[2], t0, MPI_COMM_WORLD, &stat[0]);

    MPI_Recv(getSt, 1, struct_type, ranks[0], t1, MPI_COMM_WORLD, &stat[1]);

    MPI_Wait(&req[0], &stat[0]);
    MPI_Wait(&req[1], &stat[1]);

}

// Classic Discretization wrapper.
double classicWrapper(states **state, int *tstep)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    states putSt[2];
    states getSt[2];
    int t0, t1;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        cout << "Classic Decomposition GPU" << endl;
        cudaTime timeit;
        const int xc = cGlob.xcpu/2;
        int xcp = xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;

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

        cout << "Entering Loop" << endl;

        while (t_eq < cGlob.tf)
        {
            //TIMEIN;
            classicStep<<<cGlob.gBks, cGlob.tpb>>> (dState, tmine);
            //TIMEOUT;
            classicStepCPU(state[0], xcp, tmine);
            classicStepCPU(state[2], xcp, tmine);

            cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
            cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
            cudaMemcpyAsync(state[2], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4);

            putSt[0] = state[0][1];
            putSt[1] = state[2][xc];
            classicPass(&putSt[0], &getSt[0], tmine);

            if (cGlob.bCond[0]) state[0][0] = getSt[0];
            if (cGlob.bCond[1]) state[2][xcp] = getSt[1];

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                writeOut(state, t_eq);
                twrite += cGlob.freq;
            }
        }

        cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);
        // cout << ranks[1] << " ----- " << timeit.avgt() << endl;
        //WATOM;

        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);

        cudaFree(dState);
    }
    else
    {
        int xcp = cGlob.xcpu + 1;
        mpiTime timeit;
        while (t_eq < cGlob.tf)
        {
            //TIMEIN;
            classicStepCPU(state[0], xcp, tmine);
            //TIMEOUT;

            putSt[0] = state[0][1];
            putSt[1] = state[0][cGlob.xcpu];
            classicPass(&putSt[0], &getSt[0], tmine);

            if (cGlob.bCond[0]) state[0][0] = getSt[0];
            if (cGlob.bCond[1]) state[0][xcp] = getSt[1];

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            if (t_eq > twrite)
            {
                writeOut(state, t_eq);
                twrite += cGlob.freq;
            }
        }
        // cout << ranks[1] << " ----- " << timeit.avgt() << endl;
        //WATOM;
    }
    *tstep = tmine;
	std::cout << ranks[1] << " " << printout(state[0], 0) << std::endl;

    return t_eq;
}
