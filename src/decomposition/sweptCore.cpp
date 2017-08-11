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
void wholeDiamondCPU(states *state, int tid, int zi=1, int zf=cGlob.tpbp;)
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


void passSwept(states *state, int idxend, int rnk, int offs, int offr)
{
    MPI_Isend(state + offs, cGlob.htp, struct_type, ranks[2*rnk], TAGS(tstep),
            MPI_COMM_WORLD, &req[rnk]);

    MPI_recv(state + offr, cGlob.htp, struct_type, ranks[2*rnk], TAGS(tstep), 
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
}

double sweptWrapper(states *state, double *xpts, int *tstep)
{
    cout << "Swept Decomposition" << endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        int rnk = 0, rrnk; // Seems cheesy.
        const int xc =cGlob. xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = xg+1, xgpp = xg+2;
        const size_t gpusize =  szState * xgpp;
        const size_t cpuzise = szState * xcpp;
        const size_t smem = szState * cGlob.base;
        const int offSend[2] = {1, xc}; // {left, right}    
        const int offRecv[2] = {xcp, 0}; 
        const int bks = cGlob.xg/cGlob.tpb;

        states *dState; 
        cudaMalloc((void **)&dState, gpusize);

        cudaMemcpy(dState, state, gpusize, cudaMemcpyHostToDevice);

        // Separate these.
        upTriangle <<<bks, cGlob.tpb, smem>>> (dState, tstep);

        #pragma parallel num_threads(nThreads)
        // for nWaves then for nthreads
        upTriangleCPU(state1);
        upTriangleCPU(state2);

        cudaStream_t st1, st2, st3 st4;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);
        cudaStreamCreate(&st4);

        while(t_eq < t_end)
        { 
            rrnk = rnk & 1;
            passSwept(state, xcp, rrnk, offSend[rrnk], offRecv[rrnk]);
            
            wholeDiamond <<<bks, cGlob.tpb, smem>>> (dState, tstep);

            cudaMemcpyAsync(h_left, d0_left, tpb * , cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);

            cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);
            cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st4);

            // Increment Counter and timestep
            tstep += cGlob.tpb;
            t_eq += cGlob.dt * (tstep/NSTEPS);

            // Automatic synchronization with memcpy in default stream

            if (t_eq > twrite)
            {
                downTriangle <<<bks, cGlob.tpb, smem>>> (d_IC, d2_right, d2_left);

                cudaMemcpy(T_f, d_IC, gpusize, cudaMemcpyDeviceToHost);

                upTriangle <<<bks, cGlob.tpb, smem>>> ();

                // SPLIT DIAMOND!

                twrite += freq;
            }
        }

        cudaFreeHost(h_right);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);
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
                for (int k=1; k<xcp; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);
                twrite += freq;
            }
            
        }
        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state[k]->Q[0], xpts[k], t_eq);
    }
    return t_eq;
    }

    return t_eq;

}


