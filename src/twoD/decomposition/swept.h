/*
---------------------------
    SWEPT CORE
---------------------------
*/

struct sweptConst
{
    int sideSmem, rBase, hBridgeOffset, vBridgeOffset, sharedOffset, splitOffset;
};

struct passIndex
{
    //Edges to pass;
    const int side, base;
    int nPass, sizePass;
    int *pyramid, *bridge;
    int **pyr[4];
    int **brg[4];
    
    passIndex(int side, int base) : side(side), base(base), nPass(nPass)
    {
        nPass       = side/4 * (side + 2) + (side + 2);
        sizePass    = nPass * sizeof(states);
        pyramid     = new int[4 * nPass];
        bridge      = new int[4 * nPass];
        for (int k=0; k<4; k++)
        {
            pyr[k]  = (int *) (pyramid + nPass*k);
            brg[k]  = (int *) (pyramid + nPass*k);  
        }

        this->initialize()
    }
    ~passIndex()
    {
        delete[] pyramid;
        delete[] bridge;
    }
    void initialize()
    {
        int ca[4] = {0};
        int flatidx;
        // Pyramid Loop
        for (int ky = 1; ky<=side; ky++)
        {
            for (int kx = 1; kx<=side; kx++)
            {
                flatidx = ky * base + kx;
                if (x <= y){
                    if (x+y <= side+1) pyr[1][ca[1]++] = flatidx;
                    if (x+y >= side+1) pyr[2][ca[2]++] = flatidx;
                }
                if (x >= y){
                    if (x+y <= side+1) pyr[0][ca[0]++] = flatidx;
                    if (x+y >= side+1) pyr[3][ca[3]++] = flatidx;
                }
            }
        }
        // Bridge Loop
        for (int i=0; i<3; i++) ca[i] = 0;
        for (int ky = 0; ky<=side; ky++)
        {
            for (int kx = 0; kx<=side; kx++)
            {
                flatidx = ky * base + kx;
                if (x <= y){
                    if (x+y <= side) brg[1][ca[1]++] = flatidx;
                    if (x+y >= side) brg[2][ca[2]++] = flatidx;
                }
                if (x >= y){
                    if (x+y <= side) brg[0][ca[0]++] = flatidx;
                    if (x+y >= side) brg[3][ca[3]++] = flatidx;
                }
            }
        }
    }
};

__constant__ sweptConst dSweptConst;
sweptConst hSweptConst;

__global__ void horizontalBridge(states **state, const int ts);
__global__ void verticalBridge(states **state, const int ts);
__global__ void wholeOctahedron(states **state, const int ts);

// Can it work for host too?
__device__ int 
upPyramid(states *state, int ts, const int tol, const int base)
{
    int sid;
    int kx, ky, kt;
    for (kt = 1; kt<=tol; kt++)
    {
        for (ky = threadIdx.y + kt; ky<(base - kt); ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<(base - kt); kx+=blockDim.x)
            {       
                sid =  ky * dSweptConst.rBase  + kx;
                stepUpdate(state, sid, ts++);
            }
        }    
        __syncthreads();
    }
    return ts; 
}

__device__ int 
downPyramid(states *state, int ts, const int tol, const int base)
{
    int sid;
    int kx, ky, kt;
    for (kt=tol; kt>0; kt--)
    {
        for (ky = threadIdx.y + kt; ky<(base - kt); ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<(base - kt); kx+=blockDim.x) 
            {
                sid =  ky * dSweptConst.rBase + kx;
                stepUpdate(state, sid, ts++);
            }
        }    
        __syncthreads();
    }
    return ts; 
}

__device__ void
swapMem(states *toState, states *fromState, const bool to_global)
{
    int tid, fid, tbase, fbase, toff, foff;
    if (to_global)
    {
        fbase   = dSweptConst.sideSmem;
        tbase   = dSweptConst.rBase;
        toff    = dSweptConst.sharedOffset;
        foff    = 0;
    }
    else
    { 
        fbase   = dSweptConst.rBase;
        tbase   = dSweptConst.sideSmem;
        toff    = 0;
        foff    = dSweptConst.sharedOffset;
    }

    for (ky = threadIdx.y; ky<=dSweptConst.sideSmem; ky+=blockDim.y)
    {
        for (kx = threadIdx.x; kx<=dSweptConst.sideSmem; kx+=blockDim.x) 
        {
            fid             = ky * fbase + kx + foff;
            tid             = ky * tbase + kx + toff
            toState[tid]    = fromState[fid];
        }
    }    
    __syncthreads();
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
    
    // Down pyr shared -> global.
    if (type-1)
    {   
        swapMem(sharedState, blkState, false);
        ts = downPyramid(sharedState, ts, dSweptConst.sideSmem/2, dSweptConst.sideSmem);
        swapMem(blkState, sharedState, true);
        ts = downPyramid(blkState, ts, dSweptConst.sharedOffset, dSweptConst.rBase);
    }
    if (type)
    {
        ts = upPyramid(blkState, ts, dSweptConst.sharedOffset, dSweptConst.rBase);
        swapMem(sharedState, blkState, false);
        ts = upPyramid(sharedState, ts, dSweptConst.sideSmem/2, dSweptConst.sideSmem);
        swapMem(blkState, sharedState, true);
    }
}

// It probably needs another parameter for the offset case
__global__ void homeBridges(states **regions, int ts)
{
    //Launch 1D grid of 2d Blocks
    states *horizBridge = ((states *) regions[blockIdx.x]) +  dSweptConst.hBridgeOffset; 
    states *vertBridge = ((states *) regions[blockIdx.x]) +  dSweptConst.vBridgeOffset; 
    // extern __shared__ states sharedState[];
    int sidh, sid2, dimStart;
    int rStart = (DCONST.regionSide - 2)/2;
    int kx, ky, kt;
    for (kt=rStart; kt>0; kt--)
    {
        dimStart = 1 + (rStart-kt);
        for (ky = threadIdx.y + dimStart; ky<(DCONST.regionBase - kt); ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<(DCONST.regionSide - kt); kx+=blockDim.x) 
            {
                sidh =  ky * dSweptConst.rBase + kx;
                stepUpdate(horizBridge, sidh, ts++);
                sidv =  kx * dSweptConst.rBase + ky;
                stepUpdate(vertBridge, sidv, ts++);
            }
        }    
        __syncthreads();
    }
}

/*
    MARK : HOST SWEPT ROUTINES
*/
// USE SPLIT FLAG AND OFFSET OR APPLY OFFSET IN MAIN?
void wholeOctahedronCPU(states *state, int tnow, const int type)
{
    int h, x, y;
    int sid;
    int fromEnd;
    if (type-1)
    {   
        for (h=cGlob.ht, h>0; h--)
        {
            fromEnd=cGlob.base-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y*cGlob.fullStep + x;
                    stepUpdate(state, sid, tnow++);
                }
            }
            tnow++;
        }
    }
    if (type)
    {
        for (h=1, h<cGlob.ht; h++)
        {
            fromEnd=cGlob.base-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y*cGlob.fullStep + x;
                    stepUpdate(state, sid, tnow++);
                }
            }
            tnow++;
        }
    }
}

void homeBridgeCPU(states *region, int ts)
{
    //Launch 1D grid of 2d Blocks
    states *horizBridge = ((states *) regions +  hSweptConst.hBridgeOffset); 
    states *vertBridge = ((states *) regions +  hSweptConst.vBridgeOffset); 
    // extern __shared__ states sharedState[];
    int sidh, sid2, dimStart;
    int rStart = (HCONST.regionSide - 2)/2;
    int kx, ky, kt;
    for (kt=rStart; kt>0; kt--)
    {
        dimStart = 1 + (rStart-kt);
        for (ky = dimStart; ky<(HCONST.regionBase - kt); ky++)
        {
            for(kx = kt; kx<(HCONST.regionSide - kt); kx++) 
            {
                sidh =  ky * hSweptConst.rBase + kx;
                stepUpdate(horizBridge, sidh, ts++);
                sidv =  kx * hSweptConst.rBase + ky;
                stepUpdate(vertBridge, sidv, ts++);
            }
        }    
        __syncthreads();
    }
}
/*
    MARK : PASSING ROUTINES
*/

// I THINK WE NEED ANOTHER BRIDGE ROUTINE BUT AS FAR AS I CAN TELL IT'S DONE TO HERE.
__global__
void passGPU(states *put, states *get, const int *idxmaps)
{
    const int putidx    = //(getidx+2) & 3;
    const int gid       = 1 + threadIdx.x + blockDim.x * blockIdx.x; 
    
    if (gid>A.regionSide) return;
}

__global__
void collectGPU()
{

}

void collectCPU()
{

}

int smemsize()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.sharedMemPerBlock;
}

void sweptWrapper(std::vector <Region *> &regionals)
{
    if (!rank) std::cout << " - SWEPT Decomposition - " << nproc << std::endl;

    const int gpuRegions = cGlob.hasGpu * regionals.size();
    states **regionSelector;
    
    int bigBuffer = (cGlob.regionSide * (cGlob.regionSide + 2)) / 4 + cGlob.regionBase;
    for (auto r: regionals) r->makeBuffers(buffersize, 1);
    dim3 tdim(cGlob.tpbx,cGlob.tpby);
    const int minLaunch = cGlob.regionSide/1024 + 1;
    int stepNow = regionals[0]->tStep;

    size_t smem                 = smemsize();
    int nSmem                   = smem/sizeof(states);
    int sharedCoord             = (cGlob.regionSide + )
    hSweptConst.rBase           = regionals[0].fullSide;
    hSweptConst.sideSmem        = (std::sqrt(nSmem)/2) * 2;
    smem                        = hSweptConst.sideSmem * hSweptConst.sideSmem * sizeof(states);
    hSweptConst.splitOffset     = cGlob.ht * (hSweptConst.rBase + 1);
    hSweptConst.vBridgeOffset   = hSweptConst.rBase * cGlob.htp + 1;
    hSweptConst.hBridgeOffset   = hSweptConst.rBase + cGlob.htp;
    int sharedCoord             = (cGlob.regionSide - hSweptConst.sideSmem)/2;
    hSweptConst.sharedOffset    = sharedCoord * (hSweptConst.rBase + 1);

    passIndex pidx(cGlob.regionSide, hSweptConst.rBase);

    int *dPyramid, *dBridge;
    int passAlloc = 4 * sizeof(int) * pidx.nPass;
    cudaMalloc((void **) &dPyramid, passAlloc);
    cudaMalloc((void **) &dBridge, passAlloc);
    cudaMemcpy(pidx.&pyramid, dPyramid, passAlloc, cudaMemcpyHostToDevice);
    cudaMemcpy(pidx.&bridge, dBridge, passAlloc, cudaMemcpyHostToDevice);

    if (gpuRegions) 
    {
        cudaMemcpyToSymbol(dSweptConst, &hSweptConst, sizeof(sweptConst));
        cudaMalloc((void **) &regionSelector, sizeof(states *) * gpuRegions);
        for (int i=0; i<gpuRegions; i++)
        {
            setgpuRegion <<< 1, 1 >>> (regionSelector, regionals[i]->dState, i);
        }
    }
    stepNow = regionals[0]->tStep;

    // UP_PYRAMID
    if (gpuRegions)     wholeOctahedron <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow, false, 1);
    else                wholeOctahedronCPU(regionals[0]->state, stepNow, 1);

    // PASSING SW
 
    // Then BRIDGES NE - .
    if (gpuRegions)     homeBridge <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                homeBridgeCPU (regionals[0]->state, stepNow);

    // PASSING INTERIOR EDGES - .

    while (regionals[0]->tStamp < cGlob.tf)
    {   
        // OFFSET WHOLE OCTAHEDRON
        if (gpuRegions)     wholeOctahedron <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow, false, 2);
        else                wholeOctahedronCPU(regionals[0]->state + hSweptConst.splitOffset, stepNow, 2);

        // PASSING NE

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
