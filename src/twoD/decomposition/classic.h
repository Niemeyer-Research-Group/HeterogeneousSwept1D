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

__global__ void classicStep(states **state, states **outMail, const int ts)
{
    const int gidx = blockDim.x * blockIdx.x + threadIdx.x + 1; 
    const int gidy = blockDim.y * blockIdx.y + threadIdx.y + 1; 
    stepUpdate(state, gidx, gidy, ts);
    __syncthreads();
    if (gidy == 1) outMail[0][gidx-1] = state[gidy][gidx]; // SOUTH
    if (gidx == 1) outMail[1][gidy-1] = state[gidy][gidx]; // WEST
    if (gidy == (gridDim.y * blockDim.y)) outMail[2][gidx-1] = state[gidy][gidx]; //NORTH
    if (gidx == (gridDim.x * blockDim.x)) outMail[3][gidy-1] = state[gidy][gidx]; //EAST
}

void classicStepCPU(states **state, states **outMail, const int ts)
{
    for (int k=1; k<=cGlob.yPointsRegion; k++)
    {
        for (int i=1; i<=cGlob.xPointsRegion; i++)
        {
            stepUpdate(state, i, k, ts);

            if (k == 1) outMail[0][i-1] = state[k][i];
            if (i == 1) outMail[1][k-1] = state[k][i];
            if (k == cGlob.yPointsRegion) outMail[2][i-1] = state[k][i];
            if (i == cGlob.xPointsRegion) outMail[3][k-1] = state[k][i];
        }
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
        const int gpusize = cGlob.szState * xgpp;

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

    return t_eq;
}
