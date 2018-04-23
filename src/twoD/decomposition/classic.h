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
    int kx = threadIdx.x + 1; 
    int ky = threadIdx.y + 1;
    int sid;

    for (; ky<=A.regionSide; ky+=blockDim.y)
    {
        for(; kx<=A.regionSide; kx+=blockDim.x) 
        {
            sid =  ky * A.regionBase + kx;
            stepUpdate(blkState, sid, ts);
        }
    }
}

__global__
void classicGPUSwap(states *ins, states *outs, int type)
{
    const int gid = 1 + threadIdx.x + blockDim.x * blockIdx.x; 
    
    if (gid>A.regionSide) return;

    int inidx, outidx;
    if (type & 1) 
    {
        if (type >> 1)
        {
            inidx = ((gid + 1) * A.regionBase) - 2;
            outidx = (gid * A.regionBase);
        }
        else
        {
            inidx = (gid * A.regionBase) + 1;
            outidx = ((gid + 1)  * A.regionBase) - 1;
        }
    }
    else
    {
        if (type >> 1)
        {
            inidx = gid + (A.regionSide * A.regionBase);
            outidx = gid;
        }
        else
        {
            inidx = gid + A.regionBase;
            outidx = gid + ((A.regionSide + 1) * A.regionBase);
        }
    }
    outs[outidx] = ins[inidx];
}

void classicStepCPU(states *state, const int ts)
{
    static int ky = 1;
    static int kx = 1;
    for (; ky<=A.regionSide; ky++)
    {
        for(; kx<=A.regionSide; kx++) 
        {
            sid =  ky * A.regionBase + kx;
            stepUpdate(state, sid, ts);
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
