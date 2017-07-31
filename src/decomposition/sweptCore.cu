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
    int fidx = blockDim.x + 1; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid+2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx <= (fidx-k) && tidx >= k)
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
    int fidx = blockDim.x + 1; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid + 2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx <= (fidx-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}

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


void upTriangleCPU(states *state)
{
    for (int k=1; k<ht[1]; k++)
    {
        for (int n=k, n<bsae-k, n++)
        {
            stepUpdate(state, n, tstep);
        }
    }
}


void downTriangleCPU(states *state)
{

}

void wholeDiamondCPU(states *state)
{
    for (int k=1; k<ht[1]; k++)
        for ()

    for (int k=1; k<ht[1]; k++)
    {
        for (int n=k, n<bsae-k, n++)
        {
            stepUpdate(state, n, tstep);
        }
        tstep++;
    }
}

void splitDiamondCPU(states *state)
{
    for (int k=1; k<ht[1]; k++)
        for ()

    for (int k=1; k<ht[1]; k++)
    {
        for (int n=k, n<bsae-k, n++)
        {
            stepUpdate(state, n, tstep);
        }
        tstep++;
    }
}


void passSwept(states *state, int tpb, int rank, bool dr)
{

}


//The wrapper that calls the routine functions.
double
sweptWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, const int cpu,
    states *state, states *T_f, const double freq, ofstream &fwr)
{
    const size_t devPoints = (tpb * bks + 2);
    const size_t devPortion = devPoints * sizeof(states);
    const size_t smem = (tpb + 2) * sizeof(states);
    
    const int cpuLoc = dv-tpb;

    int htcpu[5];
    for (int k=0; k<5; k++) htcpu[k] = dimz.hts[k] + 2;

	states *dState; 
    cudaMalloc((void **)&dState, sizeof(states)*dv);

	cudaMemcpy(dState, state, sizeof(states)*dv, cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;
    int tstep = 0;

	upTriangle <<<bks, tpb, smem>>> (dState, tstep);

    tstep = (SOMETHING);

    double t_eq;
    double twrite = freq - QUARTER*dt;

	// Call the kernels until you reach the final time

    REALthree *h_right, *h_left;
    REALthree *tmpr = (REALthree *) malloc(smem);
    cudaHostAlloc((void **) &h_right, tpb*sizeof(REALthree), cudaHostAllocDefault);
    cudaHostAlloc((void **) &h_left, tpb*sizeof(REALthree), cudaHostAllocDefault);

    t_eq = t_fullstep;

    cudaStream_t st1, st2, st3 st4;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);
    cudaStreamCreate(&st3);
    cudaStreamCreate(&st4);

    //Split Diamond Begin------

    wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

    cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
    cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

    cudaStreamSynchronize(st2);
    cudaStreamSynchronize(st3);

    // CPU Part Start -----

    for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

    CPU_diamond(tmpr, htcpu);

    for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

    cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
    cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

    // CPU Part End -----

    while(t_eq < t_end)
    {
        wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

        //Split Diamond Begin------

        wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

        CPU_diamond(tmpr, htcpu);

        for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
        cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

        // CPU Part End -----

        // Automatic synchronization with memcpy in default stream

        //Split Diamond End------

        t_eq += t_fullstep;

        if (t_eq > twrite)
        {
            downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

            cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

            upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

            splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

            t_eq += t_fullstep;

            twrite += freq;
        }
    }

    cudaFreeHost(h_right);
    cudaFreeHost(h_left);
    cudaStreamDestroy(st1);
    cudaStreamDestroy(st2);
    cudaStreamDestroy(st3);
    free(tmpr);

}

downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

return t_eq;

}


