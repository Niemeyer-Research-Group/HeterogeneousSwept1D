/*
---------------------------
    SWEPT CORE
---------------------------
*/

struct sweptConst
{
    int sideSmem, rBase, hBridgeOffset, vBridgeOffset, sharedOffset, splitOffset, nPass;
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

// Can it work for host too?  Yes if you guard them with preprocessor.
__device__ int 
upPyramid(states *state, int ts, const int tol, const int base)
{
    int sid;
    int kx, ky, kt;
    for (kt = 2; kt<=tol; kt++)
    {
        for (ky = threadIdx.y + kt; ky<(base - kt); ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<(base - kt); kx+=blockDim.x)
            {       
                sid =  ky * dSweptConst.rBase  + kx;
                stepUpdate(state, sid, ts, dSweptConst.rBase);
            }
        }    
        __syncthreads();
        ts++
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
                stepUpdate(state, sid, ts, dSweptConst.rBase);
            }
        }    
        __syncthreads();
        ts++;
    }
    return ts; 
}

template <bool TOGLOBAL>
__device__ void
swapMem(states *toState, states *fromState)
{
    int tid, fid, tbase, fbase, toff, foff;
    if (TOGLOBAL)
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
template <bool SPLIT, int TYPE> 
__global__ void 
wholeOctahedron(states **regions, int ts)
{
    //Launch 1D grid of 2d Blocks
    states *blkState = ((states *) regions[blockIdx.x]) + SPLIT * dSweptConst.splitOffset; 
    extern __shared__ states sharedState[];
    
    // Down pyr shared -> global.
    if (TYPE-1)
    {   
        swapMem <false> (sharedState, blkState);
        ts = downPyramid(sharedState, ts, dSweptConst.sideSmem/2, dSweptConst.sideSmem);
        swapMem <true> (blkState, sharedState);
        ts = downPyramid(blkState, ts, dSweptConst.sharedOffset, dSweptConst.rBase);
    }
    if (TYPE)
    {
        ts = upPyramid(blkState, ts, dSweptConst.sharedOffset, dSweptConst.rBase);
        swapMem <false> (sharedState, blkState);
        ts = upPyramid(sharedState, ts, dSweptConst.sideSmem/2, dSweptConst.sideSmem);
        swapMem <true> (blkState, sharedState);
    }
}
//Launch 1D grid of 2d Blocks
// It probably needs another parameter for the offset case
template <bool OFFS> 
__global__ void 
bridges(states **regions, int ts)
{
    int hoff, voff;
    if (OFFS)   
    {
        hoff = dSweptConst.hBridgeOffset; 
        voff = dSweptConst.vBridgeOffset; 
    }
    else
    {
        hoff = dSweptConst.vBridgeOffset; 
        voff = dSweptConst.hBridgeOffset; 
    }
    // extern __shared__ states sharedState[];
    int sidh, sid2, dimStart;
    int rStart = (DCONST.regionSide - 2) >> 1; 
    int kx, ky, kt;        

    states *horizBridge = ((states *) regions[blockIdx.x]) +  hoff; 
    states *vertBridge = ((states *) regions[blockIdx.x]) + voff; 

    for (kt=rStart; kt>0; kt--)
    {
        dimStart = 1 + (rStart-kt);
        for (ky = threadIdx.y + dimStart; ky<(DCONST.regionBase - kt); ky+=blockDim.y)
        {
            for(kx = threadIdx.x + kt; kx<(DCONST.regionSide - kt); kx+=blockDim.x) 
            {
                sidh =  ky * dSweptConst.rBase + kx;
                stepUpdate(horizBridge, sidh, ts, dSweptConst.rBase);
                sidv =  kx * dSweptConst.rBase + ky;
                stepUpdate(vertBridge, sidv, ts, dSweptConst.rBase);
            }
        }    
        __syncthreads();
    }
}

/*
    MARK : HOST SWEPT ROUTINES
*/
// USE SPLIT FLAG AND OFFSET OR APPLY OFFSET IN MAIN?
template <bool SPLIT, int TYPE> 
void wholeOctahedronCPU(states *state, int tnow)
{
    states *cornerState = state + SPLIT * dSweptConst.splitOffset; 

    int h, x, y;
    int sid;
    int fromEnd;
    if (TYPE-1)
    {   
        for (h=cGlob.ht, h>0; h--)
        {
            fromEnd=cGlob.base-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y*cGlob.fullStep + x;
                    stepUpdate(cornerState, sid, tnow, dSweptConst.rBase);
                }
            }
            tnow++;
        }
    }
    if (TYPE)
    {
        for (h=1, h<cGlob.ht; h++)
        {
            fromEnd=cGlob.base-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y*cGlob.fullStep + x;
                    stepUpdate(cornerState, sid, tnow, dSweptConst.rBase);
                }
            }
            tnow++;
        }
    }
}

template <bool OFFS> 
void bridgesCPU(states *regions, int ts)
{
    int hoff, voff;
    if (OFFS)
    {
        hoff = hSweptConst.hBridgeOffset; 
        voff = hSweptConst.vBridgeOffset; 
    }
    else
    {
        hoff = hSweptConst.vBridgeOffset; 
        voff = hSweptConst.hBridgeOffset; 
    }
 
    int sidh, sid2, dimStart;
    int rStart = (HCONST.regionSide - 2)/2;
    int kx, ky, kt;

    states *horizBridge = ((states *) regions + hoff); 
    states *vertBridge  = ((states *) regions + voff);
    for (kt=rStart; kt>0; kt--)
    {
        dimStart = 1 + (rStart-kt);
        for (ky = dimStart; ky<(HCONST.regionBase - kt); ky++)
        {
            for(kx = kt; kx<(HCONST.regionSide - kt); kx++) 
            {
                sidh =  ky * hSweptConst.rBase + kx;
                stepUpdate(horizBridge, sidh, ts, dSweptConst.rBase);
                sidv =  kx * hSweptConst.rBase + ky;
                stepUpdate(vertBridge, sidv, ts, dSweptConst.rBase);
            }
        }    
    }
}

/*
    MARK : PASSING ROUTINES
*/


// TYPE
// Device to Device (2) | Device to Host (1) | Host to Device (0) 
template <int LOCA, int TYPE, int OFFSET, bool INTERIOR> 
__global__
void passGPU(states *ins, states *outs, int *idxmaps)
{
    const int gid          = threadIdx.x + blockDim.x * blockIdx.x; 
    const int rOffset[4]   = {DCONST.regionSide * dSweptConst.rBase, DCONST.regionSide, -DCONST.regionSide * dSweptConst.rBase,-DCONST.regionSide};
    const int rawOffset    = OFFSET * dSweptConst.splitOffset; 

    if (gid>A.regionSide) return;    

    int inidx           = gid + (LOCA*dSweptConst.nPass);
    int outidx          = gid + (LOCA*dSweptConst.nPass);

    if (TYPE)   inidx     = idxmaps[inidx]  + rawOffset + (INTERIOR)  * rOffset[LOCA]; 
    if (TYPE-1) outidx    = idxmaps[outidx] + rawOffset + (!INTERIOR) * rOffset[LOCA];

    outs[outidx] = ins[inidx];
}

/*
InteriorPass --
EAST panel, from east neighbor, west vertical bridge:
(East idx + offset - roffset) -- INIDX
(East idx + offset) -- OUTIDX
NORTH panel, from north neighbor, south vertical bridge:
(North idx + offset - roffset) -- INIDX
(North idx + offset) -- OUTIDX
*/ 

template <int LOCA, int TYPE, int OFFSET, bool INTERIOR>
void collectCPU(states *ins, states *outs, int *idxmaps)
{
    int i;
    int workidx;
    const int rOffset[4]   = {HCONST.regionSide * hSweptConst.rBase, HCONST.regionSide, -HCONST.regionSide * hSweptConst.rBase,-HCONST.regionSide};
    const int rawOffset    = OFFSET * dSweptConst.splitOffset; 


    int inidx           = gid + (LOCA*dSweptConst.nPass);
    int outidx          = gid + (LOCA*dSweptConst.nPass);

    if (TYPE)
    {
        for (i=0; i<hSweptConst.nPass; i++)
        {
            workidx = idxmaps[inidx[i]] + rawOffset + (INTERIOR) * rOffset[LOCA];
            outs[i] = ins[workidx];
        }
    }   
    else
    {
        for (i=0; i<hSweptConst.nPass; i++)
        {
            workidx = idxmaps[outidx[i]] + rawOffset + (!INTERIOR) * rOffset[LOCA];
            outs[workidx] = ins[i];
        }
    }
}

inline int passMask(int direction)
{
    return ((direction & 3) < 2)
}

// PASSING PART --
void passPanel(std::vector <Region *> &regionals, int turn)
{
    // Last time I put the passing mechanisms in the class.  So... it won't work unless you do that or adjust the scheme.
    for (auto r: regionals)
    {
        for (auto n: r->neighbors)
        {
            if passMask(n->sidx + turn) continue;
            if (n->sameProc) //Only occurs in gpu blocks.
            {
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
            if passMask(n->ridx + turn) continue;
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
    for (auto r: regionals) r->makeBuffers(bigBuffer, 1);
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
    pidx.iniitalize();
    hSweptConst.nPass = pidx.nPass;

    int *dPyramid, *dBridge;
    //Im uncomfortable with them being the same.
    int passAlloc = 4 * sizeof(int) * pidx.nPass; 
    cudaMalloc((void **) &dPyramid, passAlloc);
    cudaMalloc((void **) &dBridge, passAlloc);
    cudaMemcpy(dPyramid, pidx.&pyramid, passAlloc, cudaMemcpyHostToDevice);
    cudaMemcpy(dBridge, pidx.&bridge, passAlloc, cudaMemcpyHostToDevice);

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
    
    // -----  UP_PYRAMID -------

    if (gpuRegions)     wholeOctahedron <false, 1> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU <false, 1> (regionals[0]->state, stepNow);

    // ----- OFFSET (SPLIT) OCTAHEDRON -------
    
    passPanel(regionals, 0);
 
    // Then BRIDGES NE - .
    if (gpuRegions)     bridges <false> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                bridgesCPU <false> (regionals[0]->state, stepNow);

    passPanel(regionals, 0);

    if (gpuRegions)     wholeOctahedron <true, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU <true, 2> (regionals[0]->state, stepNow);

    for (auto r: regionals) r->incrementTime(false); 

    while (regionals[0]->tStamp < cGlob.tf)
    {   
        // ----- HOME SLOT (WHOLE) OCTAHEDRON -------

        passPanel(regionals, 2);

        // Then BRIDGES SW - .
        if (gpuRegions)     bridges <true> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
        else                bridgesCPU <true> (regionals[0]->state, stepNow);
        
        passPanel(regionals, 2);

        if (gpuRegions)     wholeOctahedron <false, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
        else                wholeOctahedronCPU <false, 2> (regionals[0]->state, stepNow);

        for (auto r: regionals) r->incrementTime(false); 
        // ----- OFFSET (SPLIT) OCTAHEDRON -------
        
        passPanel(regionals, 0);
    
        // Then BRIDGES NE - .
        if (gpuRegions)     bridges <false> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
        else                bridgesCPU <false> (regionals[0]->state, stepNow);

        passPanel(regionals, 0);

        if (gpuRegions)     wholeOctahedron <true, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
        else                wholeOctahedronCPU <true, 2> (regionals[0]->state, stepNow);

        for (auto r: regionals) r->incrementTime(); 

    } // End Loop over all timesteps.

    // -- DOWN PYRAMID --
    // Then BRIDGES SW - .

    if (gpuRegions)     bridges <true> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                bridgesCPU <true> (regionals[0]->state, stepNow);
    
    // PASSING INTERIOR EDGES - .

    if (gpuRegions)     wholeOctahedron <false, 1> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU <false, 1> (regionals[0]->state, stepNow);

    for (auto r: regionals) r->incrementTime(false); 

    cudaFree(regionSelector);
    cudaFree(dPyramid);
    cudaFree(dBridge);
}
