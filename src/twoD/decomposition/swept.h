/*
---------------------------
    SWEPT CORE
---------------------------
*/

struct sweptConst
{
    int sideSmem, rBase, hBridgeOffset, vBridgeOffset, splitOffset;
};

sweptConst dSweptConst;
__constant__ sweptConst hSweptConst;

__global__ void horizontalBridge(states **state, const int ts);
__global__ void verticalBridge(states **state, const int ts);
__global__ void wholeOctahedron(states **state, const int ts);

// Can it work for host too?
__device__ int 
upPyramid(states *state, const int ts, const int tol, const int base)
{
    int sid;
    int kx, ky, kt;
    for (kt = 1; kt<tol; kt++)
    {
        for (ky = threadIdx.y + kt; ky<base + kt; ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<base + kt; kx+=blockDim.x) 
            {
                sid =  ky * base + kx;
                stepUpdate(state, sid, ts+kt);
            }
        }    
        __syncthreads();
    }
    return ts + kt; // Int2 to return place?  Or isn't that already tol.
}

__device__ int 
downPyramid(states *state, const int ts, const int tol, const int base)
{
    int sid;
    int kx, ky, kt;
    for (kt = 1; kt<tol; kt++)
    {
        for (ky = threadIdx.y + kt; ky>base - kt; ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; ky>base - kt; kx+=blockDim.x) 
            {
                sid =  ky * base + kx;
                stepUpdate(state, sid, ts+kt);
            }
        }    
        __syncthreads();
    }
    return ts + k; // Int2 to return place?  Or isn't that already tol
}

__device__ void
swapMem(states *toState, states *fromState, const int to_off, const int from_off)
{
    int sid;
    for (ky = threadIdx.y; ky<=dSweptConst.sideSmem; ky+=blockDim.y)
    {
        for(kx = threadIdx.x; ky<=dSweptConst.sideSmem; kx+=blockDim.x) 
        {
            sid =  ky * dSweptConst.sideSmem + kx;
            toState[sid + to_off] = fromState[sid + from_off]
        }
    }    
}



// Int not const because needs to be handled between calls.  Return with device function.
// Type 0 for downPyramid, 1 for upPyramid, 2 for wholeOcto 
// Calculate global offset as linear index as constant mem?  or 
__global__ void 
wholeOctahedron(states **regions, int ts, const bool split, const int type)
{
    //Launch 1D grid of 2d Blocks

    states *blkState = ((states *) regions[blockIdx.x]) + split * dSweptConst.splitOffset; 
    extern __shared__ states sharedState[];
    int kx = threadIdx.x + 1; 
    int ky = threadIdx.y + 1;
    
    if (type-1)
    {   
        swapMem(sharedState, blkState, 0, );
        ts = downPyramid(sharedState, ts, , dSweptConst.sideSmem);
        swapMem(blkState, sharedState, , 0);
        ts = downPyramid(blkState, ts, , dSweptConst.rBase);
    }
    if (type)
    {
        ts = upPyramid(blkState, ts, , dSweptConst.rBase);
        swapMem(sharedState, blkState, 0, );
        ts = upPyramid(sharedState, ts, , dSweptConst.sideSmem);
        swapMem(blkState, sharedState, , 0);
    }
}

__global__ void horizontalBridge(states **regions, int ts)
{
    //Launch 1D grid of 2d Blocks

    states *blkState = ((states *) regions[blockIdx.x]) + dSweptConst.hBridgeOffset; 
    extern __shared__ states sharedState[];
}
__global__ void verticalBridge(states **regions, int ts)
{
    //Launch 1D grid of 2d Blocks

    states *blkState = ((states *) regions[blockIdx.x]) +  dSweptConst.vBridgeOffset; 
    extern __shared__ states sharedState[];
}

/*
    MARK : HOST SWEPT ROUTINES
*/


void wholeOctahedronCPU(states *state, int tnow)
{
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
	tnow++;
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
    tnow++;
    }
}

void horizontalBridgeCPU()
{

}

void verticalBridgeCPU()
{
    
}


/*
    MARK : PASSING ROUTINES
*/
cout

<template T>
int smemsize(T *func)
{
    struct cudaFuncAttributes attr;
    memset(&attr, 0, sizeof(attr));
    cudaFuncGetAttributes(&attr, func);
    return attr.maxDynamicSharedSizeBytes/8;
}

void sweptWrapper(std::vector <Region *> &regionals)
{
    if (!rank) std::cout << " - SWEPT Decomposition - " << nproc << std::endl;

    const int gpuRegions = cGlob.hasGpu * regionals.size();
    states **regionSelector;
    
    int buffersize = "I've Figured this out";
    for (auto r: regionals) r->makeBuffers(buffersize, 1);
    dim3 tdim(cGlob.tpbx,cGlob.tpby);
    const int minLaunch = cGlob.regionSide/1024 + 1;
    int stepNow = regionals[0]->tStep;

    if (gpuRegions) 
    {
        cudaFuncAttributes attr;
        memset(&attr, 0, sizeof(attr));
        cudaFuncGetAttributes(&attr, func);
        const size_t smem = attr.maxDynamicSharedSizeBytes; //This needs byte size.
        hSweptConst.nSmem = smem/sizeof(states);


        cudaMemcpyToSymbol(dSweptConst, &hSweptConst, sizeof(sweptConst));
        
        cudaMalloc((void **) &regionSelector, sizeof(states *) * gpuRegions);
        for (int i=0; i<gpuRegions; i++)
        {
            setgpuRegion <<< 1, 1 >>> (regionSelector, regionals[i]->dState, i);
        }
    }

    if (gpuRegions)     wholeOctahedron <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU(regionals[0]);
    
    while (regionals[0]->tStamp < cGlob.tf)
    {   
        stepNow = regionals[0]->tStep;
        if (gpuRegions)    wholeOctahedron <<< gpuRegions, tdim >>>(regionSelector, stepNow);
        else               wholeOctahedronCPU(regionals[0]);

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


// void applyBC(states *state, int ty, int pt)
// {
//     // Like if-dirichilet
//     // Works for whole
//     state[ty*pt] = sBound[ty];
//     // If reflective
//     // state[ty*pt] = state[pt-2] or state[pt+2]
// }

