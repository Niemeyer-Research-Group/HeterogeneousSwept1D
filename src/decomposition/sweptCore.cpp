/**
    This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
*/

/*
    How to separate the equation specific EulerGlobals.h and compile the swept and MPI routines as a library?

    Remember you need to carry extra values the two edges.
    DID WITH AN EXTRA GLOBAL READ.

    Is it possible to do this without constant memory in this file.  That is, preprocessor and launch bounds only.   YEP
*/

#include "sweptCore.h"

/**
    Builds an upright triangle using the swept rule.

    Upright triangle using the swept rule.  This function is called first using the initial conditions or after results are read out using downTriange.  In the latter case, it takes the result of down triangle as IC.

    @param initial Initial condition.
    @param tstep The timestep we're starting with.
*/
__global__
void
upTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;

    // Using tidx as tid is kind of confusing for reader but looks valid.

	temper[tidx] = state[gid + 1];

    __syncthreads();

    #pragma unroll
	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k); 
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx]
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
*/
__global__
void
downTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid+2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid] = temper[tidx]
}


/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param inRight Array of right edges seeding solution vector.
    @param inLeft Array of left edges seeding solution vector.
*/
__global__
void
wholeDiamond(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid + 2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}

    #pragma unroll
	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx]
}

// HOST SWEPT ROUTINES

void upTriangleCPU(states *state)
{
    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k, n<(cGlob.base-k), n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}


void downTriangleCPU(states *state)
{
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
    //Apply BC
}

// Now you can call this on a split for all proc/threads except (0,0)
void wholeDiamondCPU(states *state, int tid, int zi=1)
{
    for (int k=cGlob.ht; k>1; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }

    for (int n=zi; n<zj; n++)
    {
        stepUpdate(state, n, tstep + (k-1));
    } 

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k, n<(cGlob.base-k), n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}

void splitDiamondCPU(states *state)
{
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < ht || n > htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k, n<cGlob.base-k), n++)
        {
            if (n < ht || n > htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }
}

void passSwept(states *state, int idxend, int rnk, int tstep, int offs, int offr)
{
    MPI_Isend(state + offs, cGlob.htp, struct_type, ranks[2*rnk], TAGS(tstep),
            MPI_COMM_WORLD, &req[rnk]);

    MPI_recv(state + offr, cGlob.htp, struct_type, ranks[2*rnk], TAGS(tstep), 
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
}

double sweptWrapper(states **state, double **xpts, int *tstep)
{
    std::cout << "Swept Decomposition" << std::endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    states **staten = new states* [cGlob.nThreads]; 
    int ar1;
    int ptin;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        int tps = cGlob.nThreads/2; // Threads per side.
        for (int k=0; k<cGlob.nThreads; k++)
        {
            ar1 = (k/tps) * 2;
            ptin = (k % tps) * cGlob.tpb; 
            staten[k] = (state[ar1] + ptin); // ptr + offset
        }

        int stride = tps * cGlob.tpb;
        int tid, strt, rnk = 0; // Seems cheesy.
        const int xc =cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const size_t gpusize = cGlob.szState * (xc + cGlob.htp);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;
        const int offSend[2] = {1, xc}; // {left, right}    
        const int offRecv[2] = {xcp, 0}; 

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);

        states *dState; 
        cudaMalloc((void **)&dState, gpusize);

        cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice);

        // Separate these.
        upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, tstep);

        // The pointers are already strided, now we stride each wave of pointers.
        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
        {
            tid = omp_get_thread_num();
            for (int k=0; k<cGlob.nWaves; k++)
            {
                strt = k*stride;
                upTriangleCPU(staten[tid] + strt, *tstep);
            }
        }

        while(t_eq < t_end)
        { 
            
            passSwept(state, xcp, rrnk, offSend[rrnk], offRecv[rrnk]);
            cudaMemcpyAsync(state[2] + offRecv[rrnk], dState, passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState + xgp, state[2] + offSend[rrnk], passsize, cudaMemcpyHostToDevice, st2);

            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState+cGlob.htp, *tstep);

            //Somehow ADD SPLIT IN HERE>

            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (cGlob.htp, *tstep);

            rnk++;
            rrnk = rnk&1;
            passSwept(state, xcp, rrnk, offSend[rrnk], offRecv[rrnk]);
            cudaMemcpyAsync(state[0] + offRecv[rrnk], passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState + xgp, state[0] + offRecv[rrnk], passsize, cudaMemcpyHostToDevice, st2);

            // Increment Counter and timestep
            tstep += cGlob.tpb;
            t_eq += cGlob.dt * (*tstep/NSTEPS);
            rnk++;

            // Automatic synchronization with memcpy in default stream

            if (t_eq > twrite)
            {
                downTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, *tstep);

                cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);
                                
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);                
                for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
                for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

                upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, *tstep);

                // SPLIT DIAMOND!

                twrite += cGlob.freq;
            }
        }

        cudaFree(dState);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
    }

    else
    {
        int xcp = cGlob.xcpu + htp;
        upTriangleCPU(state);

        while (t_eq < t_end)
        {
            passSwept(state, xcp); // Left first

            if (rank == 0 && )
            {
                splitDiamondCPU(state + cGlob.htp)
            }
            else
            {
                wholeDiamondCPU(state + cGlob.htp, cGlob.tpb);
            }

            passSwept(state, xcp); // Then Right

            if (t_eq > twrite)
            {
                #pragma omp parallel for
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
                twrite += cGlob.freq;
            }
            
        }
        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
    }

    return t_eq;
}


