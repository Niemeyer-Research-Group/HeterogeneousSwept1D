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



// KERNELS //
#include <algorithm>

__global__ void 
classicStep(states **regions, const int ts)
{
    states *blkState = (states *) regions[blockIdx.x]; //Launch 1D grid of 2d Blocks
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

// This isn't a two way swap yet is it?
__global__ void 
classicGPUSwap(states *ins, states *outs, const int loc, const int type)
{
    const int gid = 1 + threadIdx.x + blockDim.x * blockIdx.x; 
    
    if (gid>A.regionSide) return;

    int inidx   = (gid-1) + (loc*A.regionSide);
    int outidx  = (gid-1) + (loc*A.regionSide);

    // if both (2) it's device to device 
    if (type) //Device to Host (1)
    {
        if (loc & 1) // If odd its on the veritcal edge (1, 3)
        {
            if (loc >> 1)   inidx = ((gid + 1) * A.regionBase) - 2; //(3)
            else            inidx = (gid * A.regionBase) + 1; //(1)
        }
        else // It's even, its horizontal.
        {
            if (loc >> 1)   inidx = gid + (A.regionSide * A.regionBase); //(2)
            else            inidx = gid + A.regionBase; //(0)
        }
    }
    if (type-1) //Host to Device (0)
    {
        if (loc & 1) // (1 , 3)
        {
            if (loc >> 1)   outidx = (gid * A.regionBase);
            else            outidx = ((gid + 1)  * A.regionBase) - 1;
        }
        else    // (2, 4)
        {
            if (loc >> 1)   outidx = gid;
            else            outidx = gid + ((A.regionSide + 1) * A.regionBase);
        }
    }
    outs[outidx] = ins[inidx];
}

void classicStepCPU(Region *regional)
{
    int ky, kx;
    int sid;
    for (ky = 1; ky<=A.regionSide; ky++)
    {
        for(kx = 1; kx<=A.regionSide; kx++) 
        {
            sid =  ky * A.regionBase + kx;
            stepUpdate(regional->state, sid, regional->tStep);
        }
    }
}

void cpuBufCopy(states **stRows, states *buf, const int loc, const int type)
{
    int i;
    const int offs  = cGlob.regionSide * loc;

    if (type)
    {
        switch(loc)
        {
            case 0: for (i=0; i<cGlob.regionSide; i++) 
                buf[i+offs] = stRows[1][i+1];
            case 1: for(i=0; i<cGlob.regionSide; i++)
                buf[i+offs] = stRows[1+i][1];
            case 2: for(i=0; i<cGlob.regionSide; i++)
                buf[i+offs] = stRows[cGlob.regionSide][i+1];
            case 3: for(i=0; i<cGlob.regionSide; i++) 
                buf[i+offs] = stRows[i+1][cGlob.regionSide];
        }
    }
    else
    {
        switch(loc)
        {
            case 0: for(i=0; i<cGlob.regionSide; i++) 
                stRows[0][i+1] = buf[i+offs];
            case 1: for(i=0; i<cGlob.regionSide; i++) 
                stRows[i+1][0] = buf[i+offs];
            case 2: for(i=0; i<cGlob.regionSide; i++) 
                stRows[cGlob.regionSide+1][i+1] = buf[i+offs];
            case 3: for(i=0; i<cGlob.regionSide; i++) 
                stRows[i+1][cGlob.regionSide+1] = buf[i+offs];
        }
    }
}

// Classic Discretization wrapper.
void classicWrapper(std::vector <Region *> &regionals)
{
    if (!rank) std::cout << " - CLASSIC Decomposition - " << nproc << std::endl;
    const int gpuRegions = cGlob.hasGpu * regionals.size();
    states **regionSelector;
    if (gpuRegions) 
    {
        cudaMalloc((void **) &regionSelector, sizeof(states *) * gpuRegions);
        for (int i=0; i<gpuRegions; i++)
        {
            setgpuRegion <<< 1, 1 >>> (regionSelector, regionals[i]->dState, i);
         }
    }

    
    for (auto r: regionals) r->makeBuffers(cGlob.regionSide, 4);
    dim3 tdim(cGlob.tpbx,cGlob.tpby);
    const int minLaunch = cGlob.regionSide/1024 + 1;
    int stepNow;

    while (regionals[0]->tStamp < cGlob.tf)
    {   
        stepNow = regionals[0]->tStep;
        if (gpuRegions)     classicStep <<< gpuRegions, tdim >>>(regionSelector, stepNow);
        else                classicStepCPU(regionals[0]);

        for (auto r: regionals)
        {
            for (auto n: r->neighbors)
            {
            
                if (n->sameProc) //Only occurs in gpu blocks.
                {
                    n->printer();
                    classicGPUSwap <<< minLaunch, cGlob.regionSide >>> (r->dState, regionals[n->id.localIdx]->dState, n->sidx, 2);
                }
                else
                {
                    if (gpuRegions) 
                    {
                        classicGPUSwap <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, n->sidx, 1);
                        r->gpuBufCopy(1, n->sidx);
                    }
                    else
                    {
                        cpuBufCopy(r->stateRows, r->sendBuffer, n->sidx, 1);
                        r->bufMessage(1, n->sidx);
           
                    }
                }
            }
        }
        // RECEIVE
        for (auto r: regionals) 
        {
            r->incrementTime(); //Any Write out occurs within the swapping procedure.
            for (auto n: r->neighbors)
            {
                if(!n->sameProc)
                {
                    if (gpuRegions) 
                    {   
                        r->gpuBufCopy(0, n->sidx);
                        classicGPUSwap <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dRecv, n->sidx, 0);
                    }
                    else
                    {
        
                        r->bufMessage(0, n->sidx);
                        cpuBufCopy(r->stateRows, r->recvBuffer, n->sidx, 0);
                    } // End gpu vs cpu receiver choice. 
                }     // End mask over neighbors already swapped. 
            }         // End Loop over all this region's neighbors. 
        }             // End Loop over all regions on this process. 
    }                 // End while loop over all timesteps
    cudaFree(regionSelector);
}

