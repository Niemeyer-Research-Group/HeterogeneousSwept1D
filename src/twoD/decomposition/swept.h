/*
---------------------------
    SWEPT CORE
---------------------------
*/
#include <helper_cuda.h>
#include <vector>
#include <array>
#include <algorithm>
#include <iterator>
// IF assign() proves necessary #include<algorithm>

struct sweptConst
{
    int sideSmem, rBase, hBridgeOffset, vBridgeOffset, sharedOffset, splitOffset, nPass;
};

struct PassIndex
{
    //Edges to pass;
    typedef std::vector<int> idxvec;
    const int side, base, gpu;
	idxvec triangles;
    intvec passer;
    int nPass, sizePass;
    int cnt;

    PassIndex(int side, int base, int hasG) : side(side), base(base), gpu(cGlob.hasGpu), triangles(initialize())
    {
        nPass       = side/4 * (side + 2) + (side + 2);
        sizePass    = 4 * nPass * sizeof(int);
        cnt = 0;
    }

    PassIndex(const PassIndex &pix) : side(pix.side), base(pix.base), gpu(pix.gpu), sizePass(pix.sizePass),
				nPass(pix.nPass), pyramid(pix.pyramid), bridge(pix.bridge){   }


	template <int TYPE, bool OFFSET, bool INTERIOR>
	void passCPU(states *ins, states *outs, const int loca);

	template <bool OFFSET, bool INTERIOR>
    void passPanel(std::vector <Region *> &regionals);

    idxvec initialize()
    {
		idxvec starter;
        int flatidx;
        // Pyramid Loop
        for (int ky = 0; ky<side+2; ky++)
        {
            for (int kx = ky; kx<(side-ky+2); kx++)
            { 
                flatidx = ky * base + kx
                starter.push_back(flatidx);
            }
       	}
		return starter;
    }

    int getRot(int dir, int idxx)
    {
        pass; // Something something
    }

    void ppass(int extent=10)
    {
        for (int i=0; i<extent; i++) std::cout << pyramid[1][i] << " ";
        std::cout << std::endl;
    }

    // int *getPtr(const bool INTERIOR)
    // {
    //     if (gpu) {
    //         if (INTERIOR) return dBridge;
    //         else          return dPyramid;
    //     }
    //     else {
    //         if (INTERIOR) return bridge;
    //         else          return pyramid;
    //     }
    // }

    // void plotPass(int eye);
};

// #include "matplotlib-cpp/matplotlibcpp.h"

// namespace plt = matplotlibcpp;

// void PassIndex::plotPass(int eye)
// {
//     std::vector<size_t> px, py;
//     int neweye, cnew;
//     std::string rc[] = {".r", ".b", ".g", ".k"};
//     std::string titles[] = {"South", "East", "North", "West"};
//     for (int i=0; i<4; i++)
//     {
//         cnew = 0;
//         for (int k=0; k<nPass; k++)
//         {
//             if (eye){
//                 px.push_back(pyramid[i][k] % base);
//                 py.push_back(pyramid[i][k] / base);
//             }
//             else{
//                 px.push_back(bridge[i][k] % base);
//                 py.push_back(bridge[i][k] / base);
//             }
//         }

//         plt::named_plot( titles[i], px, py,rc[i]);
//         std::cout << "Im Plotting!" << px.size() << std::endl;
//         neweye = px.back();
//         px.pop_back();

//         while (neweye == 0){
//             std::cout << i << " " << neweye << "   ";
//             neweye=px.back();
//             px.pop_back();
//             cnew++;
//         }
//         std::cout << "---------" << cnew << std::endl;

//         px.clear();
//         py.clear();
//     }
//     plt::title("PyramidPass");
//     plt::legend();
//     plt::show();
//     std::cout << "Im Plotting!" << px.size() << std::endl;
// }

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
        ts++;
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
    int tid, fid, tbase, fbase, toff, foff, kx, ky;
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
            tid             = ky * tbase + kx + toff;
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
template <bool OFFSET>
__global__ void
bridges(states **regions, int ts)
{
    const int rStart = ((DCONST.regionSide - 2) >> 1);
    int hoff, voff;
    if (OFFSET)
    {
        hoff = dSweptConst.hBridgeOffset;
        voff = dSweptConst.vBridgeOffset;
    }
    else
    {
        hoff = dSweptConst.vBridgeOffset;
        voff = dSweptConst.hBridgeOffset;
    }

    int sidh, sidv, dimStart;
    int kx, ky, kt;

    states *horizBridge = ((states *) regions[blockIdx.x]) + hoff;
    states *vertBridge = ((states *) regions[blockIdx.x])  + voff;

    for (kt = rStart; kt>0; kt--)
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
        ts++;
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
    states *cornerState = state + SPLIT * hSweptConst.splitOffset;

    int h, x, y;
    int sid;
    int fromEnd;
    if (TYPE-1)
    {
        for (h=cGlob.ht; h>0; h--)
        {
            fromEnd = cGlob.regionBase-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y * hSweptConst.rBase + x;
                    stepUpdate(cornerState, sid, tnow, hSweptConst.rBase);
                }
            }
            tnow++;
        }
    }
    if (TYPE)
    {
        for (h=1; h<cGlob.ht; h++)
        {
            fromEnd=cGlob.regionBase-h;
            for (y=h; y<fromEnd; y++)
            {
                for (x=h; x<fromEnd; x++)
                {
                    sid = y * hSweptConst.rBase + x;
                    stepUpdate(cornerState, sid, tnow, hSweptConst.rBase);
                }
            }
            tnow++;
        }
    }
}

template <bool OFFSET>
void bridgesCPU(states *regions, int ts)
{
    static const int rStart = (HCONST.regionSide - 2)/2;
    int hoff, voff;
    if (OFFSET)
    {
        hoff = hSweptConst.hBridgeOffset;
        voff = hSweptConst.vBridgeOffset;
    }
    else
    {
        hoff = hSweptConst.vBridgeOffset;
        voff = hSweptConst.hBridgeOffset;
    }

    int sidh, sidv, dimStart;
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
        ts++;
    }
}

/*
    MARK : PASSING ROUTINES
*/


// TYPE
// Device to Device (2) | Device to Host (1) | Host to Device (0) 
template <int TYPE, bool OFFSET, bool INTERIOR> 
__global__
void passGPU(states *ins, states *outs, const int *idxmaps, const int loca)
{
    const int gid          = threadIdx.x + blockDim.x * blockIdx.x; 
    const int rOffset[4]   = {DCONST.regionSide * dSweptConst.rBase, DCONST.regionSide, -DCONST.regionSide * dSweptConst.rBase,-DCONST.regionSide};
    const int rawOffset    = OFFSET * dSweptConst.splitOffset;

    if (gid > dSweptConst.nPass) return;

    int inidx           = gid + (loca*dSweptConst.nPass);
    int outidx          = gid + (loca*dSweptConst.nPass);

    if (TYPE)   inidx     = idxmaps[inidx]  + rawOffset + (INTERIOR)  * rOffset[loca];
    if (TYPE-1) outidx    = idxmaps[outidx] + rawOffset + (!INTERIOR) * rOffset[loca];

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

template <int TYPE, bool OFFSET, bool INTERIOR>
void PassIndex::passCPU(states *ins, states *outs, const int loca)
{
    static const int rOffset[4]   = {HCONST.regionSide * hSweptConst.rBase, HCONST.regionSide, -HCONST.regionSide * hSweptConst.rBase,-HCONST.regionSide};

    int i, workidx;
    int rawOffset = (int)OFFSET * hSweptConst.splitOffset;
    // std::cout << rank << " -BEFORE- " << " - " << TYPE << " - " << INTERIOR << " - "  << OFFSET << " - " << loca << std::endl;
	int roff 		= rawOffset + ((int)(INTERIOR) * rOffset[loca]);
    if (INTERIOR)
    {
        std::copy(bridge[loca].begin(), bridge[loca].end(), std::back_inserter(passer));
    }
    else
    {
        std::copy(pyramid[loca].begin(), pyramid[loca].end(), std::back_inserter(passer));
    }

    if (TYPE)
    {
        std::cout << "INCOMING: " << rank << " " << cnt << " " << roff << " " << loca << " " << INTERIOR << " " << rawOffset << " | ";
        for (i = 0; i<passer.size(); i++)
        {
            std::cout << "  " <<  pyramid[loca][i];
            workidx         = passer[i] + roff;
            outs[i]         = ins[workidx];
        }

    }
    else
    {
        std::cout << "OUTGOING: " << rank << " " << cnt << " " << roff << " " << loca << " " << INTERIOR << " " << rawOffset << " | ";
        for (i = 0; i<passer.size(); i++)
        {
            std::cout << "  " <<  pyramid[loca][i];
            workidx         = passer[i] + roff;
            outs[workidx] 	= ins[i];
        }
    
    }
    std::cout << std::endl;
    passer.clear();
    return;
}


    // std::cout << std::boolalpha << rank << " -AFTER- " << " - " << TYPE << " - " << INTERIOR << " - "  << OFFSET << " - " << workidx << " - " << loca << std::endl;


inline bool passMask(int direction)
{
    return ((direction & 3) > 1);
}

// PASSING PART --
__device__ int *dPyramid, *dBridge;

template <bool OFFSET, bool INTERIOR>
void PassIndex::passPanel(std::vector <Region *> &regionals)
{
	const int det = 2*OFFSET;
    static const int minLaunch = cGlob.regionSide/1024 + 1;

    // Last time I put the passing mechanisms in the class.  So... it won't work unless you do that or adjust the scheme.
    // std::cout << "Begin passPanel - " << cnt << " " << rank << std::endl;
    // this->ppass(100);
    // MPI_Barrier(MPI_COMM_WORLD);
    for (auto r: regionals)
    {
        for (auto n: r->neighbors)
        {
            if (passMask(n->sidx + det))
			{
            	if (n->sameProc) //Only occurs in gpu blocks.
            	{
					if (INTERIOR){
                	passGPU <2, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, regionals[n->id.localIdx]->dState, dBridge, n->sidx);
					}
                    else
                    {
					passGPU <2, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, regionals[n->id.localIdx]->dState, dPyramid, n->sidx);
					}
            	}
            	else
            	{
                	if (r->self.gpu)
               		{
                        if (INTERIOR){
                	        passGPU <1, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, dBridge, n->sidx);
					    } 
                        else 
                        {
					        passGPU <1, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, dPyramid, n->sidx);
					    }
                    	passGPU <1, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, dPyramid, n->sidx);
                    	r->gpuBufCopy(1, n->sidx);
                	}
                	else
                	{
                    	passCPU <1, OFFSET, INTERIOR> (r->state, r->sendBuffer, n->sidx);
                    	r->bufMessage(1, n->sidx);
					}
				}
			}
			else
			{
                // std::cout << "START  RECV rank - " << rank << " " << n->id.owner << " dir " << n->sidx << " " << cnt << " | (" << r->self.globalx <<  ", " << r->self.globaly << ") <- (" << n->id.globalx << ", " << n->id.globaly << ") at - " << __LINE__ << std::endl;
                if (r->self.gpu)
                {
                    r->gpuBufCopy(0, n->sidx);
                        if (INTERIOR){
                	        passGPU <0, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, dBridge, n->sidx);
					    } 
                        else 
                        {
					        passGPU <0, OFFSET, INTERIOR> <<< minLaunch, cGlob.regionSide >>> (r->dState, r->dSend, dPyramid, n->sidx);
					    }
                }
                else
                {
                    r->bufMessage(0, n->sidx);
                    passCPU <0, OFFSET, INTERIOR> (r->recvBuffer, r->state, n->sidx);
                } // End gpu vs cpu receiver choice.
            }     // End mask over neighbors already swapped.
		}
	}
    for (auto r: regionals)
    {
        MPI_Waitall(r->reqvec.size(), &r->reqvec[0], &r->statvec[0]);
		r->reqvec.clear();
		r->statvec.clear();
    }
    // std::cout << "END passPanel - " << cnt << " " << rank << std::endl;
    // this->ppass(100);


    // std::cout << cnt << " Enter a Message about ska: " << std::endl;
    // std::cin >> rudy;

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (!rudy) {
    //     endMPI();
    //     exit(0);
    // }

    cnt++;
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
    if (!rank) std::cout << std::boolalpha << " - SWEPT Decomposition - " << nprocs << std::endl;

    const int gpuRegions = cGlob.hasGpu * regionals.size();
    states **regionSelector;
    int bigBuffer = (cGlob.regionSide * (cGlob.regionSide + 2)) / 4 + cGlob.regionBase;
    for (auto r: regionals) r->makeBuffers(bigBuffer, 2);
    dim3 tdim(cGlob.tpbx,cGlob.tpby);
    const int minLaunch = cGlob.regionSide/1024 + 1;
    int stepNow = regionals[0]->tStep;
    // if (!rank) std::cout << "Made it swept0 " << rank << " " << __LINE__ << std::endl;

    size_t smem                 = smemsize();
    int nSmem                   = smem/sizeof(states);
    hSweptConst.rBase           = regionals[0]->fullSide;
    hSweptConst.sideSmem        = (std::sqrt(nSmem)/2) * 2;
	int shareCoord				= (cGlob.regionBase - hSweptConst.sideSmem)/2;
    smem                        = hSweptConst.sideSmem * hSweptConst.sideSmem * sizeof(states);
    hSweptConst.splitOffset     = cGlob.ht * (hSweptConst.rBase + 1);
    hSweptConst.vBridgeOffset   = hSweptConst.rBase * cGlob.htp + 1;
    hSweptConst.hBridgeOffset   = hSweptConst.rBase + cGlob.htp;
    int sharedCoord             = (cGlob.regionSide - hSweptConst.sideSmem)/2;
    hSweptConst.sharedOffset    = sharedCoord * (hSweptConst.rBase + 1);

    // std::cout << "Made it swept1 " << rank << " " << __LINE__ << std::endl;
    PassIndex pidx(cGlob.regionSide, hSweptConst.rBase, cGlob.hasGpu);
    hSweptConst.nPass = pidx.nPass;
    cudaMalloc((void **)& dPyramid, pidx.sizePass);
    cudaMalloc((void **)& dBridge, pidx.sizePass);
    cudaMemcpy(dPyramid, &pidx.pyramid, pidx.sizePass, cudaMemcpyHostToDevice);
    cudaMemcpy(dBridge, &pidx.bridge, pidx.sizePass, cudaMemcpyHostToDevice);
    //if(rank==1) pidx.plotPass();

    MPI_Barrier(MPI_COMM_WORLD);
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
    // std::cout << "Made it swept2 " << rank << " " << __LINE__ << std::endl;
    // -----  UP_PYRAMID -------

    if (gpuRegions)     wholeOctahedron <false, 1> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                wholeOctahedronCPU <false, 1> (regionals[0]->state, stepNow);
    cudaKernelCheck(cGlob.hasGpu);

    // ----- OFFSET (SPLIT) OCTAHEDRON -------
    MPI_Barrier(MPI_COMM_WORLD);
    pidx.passPanel <false, false> (regionals);

    // Then BRIDGES NE - .
    if (gpuRegions)     bridges <false> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                bridgesCPU <false> (regionals[0]->state, stepNow);

    pidx.passPanel <false, true> (regionals);
    // std::cout << "Made it swept " << rank << " " << __LINE__ << std::endl;

    if (gpuRegions)     wholeOctahedron <true, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU <true, 2> (regionals[0]->state, stepNow);
    // std::cout << "Made it swept " << rank << " " << __LINE__ << std::endl;
    cudaKernelCheck(cGlob.hasGpu);

    for (auto r: regionals) r->incrementTime(false);

    while (regionals[0]->tStamp < cGlob.tf)
    {
        // ----- HOME SLOT (WHOLE) OCTAHEDRON -------
        // std::cout << "TIME: " <<  regionals[0]->tStamp << " RANK: " << rank << " " << __LINE__ << std::endl;

        pidx.passPanel<true, false> (regionals);

        // Then BRIDGES SW - .
        if (gpuRegions)     bridges <true> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
        else                bridgesCPU <true> (regionals[0]->state, stepNow);

	    pidx.passPanel<true, true> (regionals);

        if (gpuRegions)     wholeOctahedron <false, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
        else                wholeOctahedronCPU <false, 2> (regionals[0]->state, stepNow);

        for (auto r: regionals) r->incrementTime(false);
        // ----- OFFSET (SPLIT) OCTAHEDRON -------

        pidx.passPanel <false, false> (regionals);
    
        // Then BRIDGES NE - .
        if (gpuRegions)     bridges <false> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
        else                bridgesCPU <false> (regionals[0]->state, stepNow);

        pidx.passPanel <false, true> (regionals);

        if (gpuRegions)     wholeOctahedron <true, 2> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
        else                wholeOctahedronCPU <true, 2> (regionals[0]->state, stepNow);

        for (auto r: regionals) r->incrementTime();

    } // End Loop over all timesteps.

    // -- DOWN PYRAMID --
    pidx.passPanel <true, false> (regionals);

    if (gpuRegions)     bridges <true> <<< gpuRegions, tdim, smem >>> (regionSelector, stepNow);
    else                bridgesCPU <true> (regionals[0]->state, stepNow);
    
    pidx.passPanel <true, true> (regionals);

    if (gpuRegions)     wholeOctahedron <false, 1> <<< gpuRegions, tdim, smem >>>(regionSelector, stepNow);
    else                wholeOctahedronCPU <false, 1> (regionals[0]->state, stepNow);


    std::cout << "ITS OVER ---- TIME: " <<  regionals[0]->tStamp << " RANK: " << rank << " " << __LINE__ << std::endl;
    for (auto r: regionals) r->incrementTime(false); 

    if (gpuRegions) cudaFree(regionSelector);
}
