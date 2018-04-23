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

struct Address;
struct Neighbor;
struct Region;

void sender(Region *home, Region *neighbor, idx)
{
    static const int items[4] = {cGlob.yPointsRegion,                                     cGlob.xPointsRegion, 
                                cGlob.yPointsRegion, cGlob.xPointsRegion};


    cudaMemcpy(home->dInbox[idx], neighbor->dOutbox[idx] cGlob.szState*items[idx], cudaMemcpyDeviceToDevice);
    // Now what to do about the gpu process getting data from the 
}

void sender(Region *home, Address *neighbor) 
{
    static const int items[4] = {cGlob.yPointsRegion, cGlob.xPointsRegion, cGlob.yPointsRegion, cGlob.xPointsRegion};
    
    MPI_ISend(home->inbox[neighbor->sidx], items[neighbor->sidx], struct_type, neighbor->id.owner, neighbor->id.localIdx, MPI_COMM_WORLD, neighbor->req); // PUT REQUEST IN NEIGHBOR OBJECT
}

void pass(std::vector <Region *> &regionals)
{
    int nidx;
    Region *rix;
    for (auto &r: regionals)
    {
        for (int i=0; i<4; i++)
        {
            if (r->neighbors[i].sameProc) 
            {
                // return bool if this process = other process
                nidx = r->neighbors[i].id.localIdx;
                rix = regionals[nidx];
                sender(r, rix, i);
            }
            else 
        }
    
    }    

}

__global__ void classicStep(states **regions, const int ts)
{
    states *blkState = regions[blockIdx.x]; //Launch 1D grid of 2d Blocks
    const int tidx = threadIdx.x + 1; 
    const int tidy = threadIdx.y + 1;

    int idx, idy, sid;

    for (int kx=0; kx<A.sBlocks; kx++)
    {
        idy = tidy + ky * blockDim.y;
        for(int ky=0; ky<A.sBlocks; ky++) 
        {
            sid = idy * A.sBlocks + kx * blockDim.x + tidx;
            stepUpdate(blkState, sid, ts);
        }
        __syncthreads();
    }
    
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

// Classic Discretization wrapper.
double classicWrapper(Region **region)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    int t0, t1;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        cout << "Classic Decomposition GPU" << endl;
        cudaTime timeit;
        const int xc = cGlob.xcpu/2;
        int xcp = xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize = cGlob.szState * xgpp;

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

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                for (auto r: region)
                {
                    r->solutionOutput(tmine);
                }
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
