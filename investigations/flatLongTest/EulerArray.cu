
/** 
    Flattening strategy for Euler - Atomic stages.

    COMPILE:
    nvcc EulerArray.cu -o ./bin/EulerArray -gencode arch=compute_35,code=sm_35 -lm -restrict -Xptxas=-v 
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "myVectorTypes.h"

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

// This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.

#ifndef GPUNUM
    #define GPUNUM              0
#endif

// We're just going to assume doubles
#define REAL            double
#define REALtwo         double2
#define REALthree       double3
#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0
#define SQUAREROOT(x)   sqrt(x)

// Hardwire in the length of the 
const REAL lx = 1.0;

// The structure to carry the initial and boundary conditions.
// 0 is left 1 is right.
REALthree bd[2];

//dbd is the boundary condition in device constant memory.
__constant__ REALthree dbd[2]; 

//Protoype for useful information struct.
struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int base; // Length of node + stencils at end (4)
    int idxend; // Last index (number of spatial points - 1)
    int idxend_1; // Num spatial points - 2
    int hts[5]; // The five point stencil around base/2
};

// structure of dimensions in cpu memory
dimensions dimz;

// Useful and efficient to keep the important constant information in GPU constant memory.
__constant__ dimensions dimens;


__host__ __device__
__forceinline__
void
readIn(REALthree *temp, const REALthree *rights, const REALthree *lefts, int td, int gd)
{
    // The index in the SHARED memory working array to place the corresponding member of right or left.
    #ifdef __CUDA_ARCH__  // Accesses the correct structure in constant memory.
	int leftidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + ((td>>2)<<1) + (td & 3);
    #else
    int leftidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + (td & 3) - (4 + ((td>>2)<<1));
    int rightidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + ((td>>2)<<1) + (td & 3);
    #endif

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}


__device__
__forceinline__
void
writeOutRight(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    int gdskew = (gd + bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2; 
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}


__host__ __device__
__forceinline__
void
writeOutLeft(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    #ifdef __CUDA_ARCH__
    int gdskew = (gd - bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
    #else
    int gdskew = gd;
    int leftidx = (((td>>2) & 1)  * dimz.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimz.base-6) + (((td>>2) & 1)  * dimz.base) + (td & 3) - ((td>>2)<<1);
    #endif

    rights[gd] = temp[rightidx];
    lefts[gdskew] = temp[leftidx];
}


__device__ __host__
__forceinline__
REAL
pressure(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #else
    return dimz.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #endif
}

__device__ __host__
__forceinline__
REAL
pressureHalf(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - HALF * current.y * current.y);
    #else
    return dimz.mgam * (current.z - HALF * current.y * current.y);
    #endif
}


__device__ __host__
__forceinline__
REALthree
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    return (cvCurrent + HALF * min(pRatio,ONE) * (cvOther - cvCurrent));
}


__device__ __host__
__forceinline__
REALthree
eulerFlux(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;

    REAL pL = pressure(cvLeft);
    REAL pR = pressure(cvRight);

    REALthree flux;
    flux.x = (cvLeft.y + cvRight.y);
    flux.y = (cvLeft.y*uLeft + cvRight.y*uRight + pL + pR);
    flux.z = (cvLeft.z*uLeft + cvRight.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}


__device__ __host__
__forceinline__
REALthree
eulerSpectral(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif

    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(cvLeft.x);
    REAL rhoRightsqrt = SQUAREROOT(cvRight.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*cvRight.y + rhoRightsqrt*cvLeft.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*cvRight.z + rhoRightsqrt*cvLeft.z)*halfDenom;

    REAL pH = pressureHalf(halfState);

    #ifdef __CUDA_ARCH__
    return (SQUAREROOT(pH*dimens.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #else
    return (SQUAREROOT(pH*dimz.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #endif
}


__device__ __host__
REALthree
eulerStutterStep(REALthree *state, int tr, char flagLeft, char flagRight)
{
    //P1-P0
    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;
    //P2-P1
    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));
    //P3-P2
    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));
    //P4-P3
    REAL pRR = (flagRight) ? ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));

    //This is the temporary state bounded by the limitor function.
    //Pr0 = PL/PLL*rho0/rho2  Pr0 is not -, 0, or nan.
    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan, pass Pr1^-1.
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    //Pressure needs to be recalculated for the new limited state variables.
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan.
    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    //Pr2 = PRR/PR*rho2/rho4  Pr2 is not - or nan, pass Pr2^-1.
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0)) ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    return state[tr] + (QUARTER * dimens.dt_dx * flux);
    #else
    return state[tr] + (QUARTER * dimz.dt_dx * flux);
    #endif

}

__device__ __host__
REALthree
eulerFinalStep(REALthree *state, int tr, char flagLeft, char flagRight)
{

    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;

    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));

    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));

    REAL pRR = (flagRight) ?  ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));

    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0))  ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    // Return only the RHS of the discretization.
    #ifdef __CUDA_ARCH__
    return (HALF * dimens.dt_dx * flux);
    #else
    return (HALF * dimz.dt_dx * flux);
    #endif

}

__global__
void
classicEuler(REALthree *euler_in, REALthree *euler_out, const bool finalstep)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    if (truth.x)
    {
        euler_out[gid] = dbd[0];
        return;
    }
    else if (truth.w)
    {
        euler_out[gid] = dbd[1];
        return;
    }

    if (finalstep)
    {
        euler_out[gid] += eulerFinalStep(euler_in, gid, truth.y, truth.z);
    }
    else
    {
        euler_out[gid] = eulerStutterStep(euler_in, gid, truth.y, truth.z);
    }
}

__global__
void
upTriangle(const REALthree *IC, REALthree *outRight, REALthree *outLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tididx = threadIdx.x + 2; //Thread's lane in the node [2, blockDim.x+1]
    int tidxTop = tididx + dimens.base; //Thread's second row lane. 
    int k=4; //Start k at 4 since the base and second step occur before loop.

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

    __syncthreads();

    //First step gets predictor values for lanes excluding outer two on each side.
	if (threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
		temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
	}

	__syncthreads();

	//Step through solution excluding two more lanes on each side each step.
	while (k < (blockDim.x>>1))
	{
		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, false, false);
		}

        k+=2;
		__syncthreads();

		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
		}

		k+=2;
		__syncthreads();

	}
    // Passes right and keeps left
    writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}

__global__
void
downTriangle(REALthree *IC, const REALthree *inRight, const REALthree *inLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    //k starts at blockDim.x/2 and shrinks from there as the lane width grows.
    int k = dimens.hts[2]; 

    //Masks edges (whole domain edges not nodal edges) on last timestep. 
    //Stored in one register per thread.
    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    readIn(temper, inRight, inLeft, threadIdx.x, gid);

    __syncthreads();

	while(k>1)
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}

        k-=2;
        __syncthreads();

        if (!truth.x && !truth.w && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

		k-=2;
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}

__global__
void
wholeDiamond(const REALthree *inRight, const REALthree *inLeft, REALthree *outRight, REALthree *outLeft)
{

    extern __shared__ REALthree temper[];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;

    //Masks edges in the same way as downTriangle
    char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    readIn(temper, inRight, inLeft, threadIdx.x, gid);

    __syncthreads();

    //k starts behind the downTriangle k because we need to do the first timestep outside the loop
    //to get the order right. 
    int k = dimens.hts[0];

    if (tididx < (dimens.base-dimens.hts[2]) && tididx >= dimens.hts[2])
    {
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
    }

    __syncthreads();

    while(k>4)
    {
        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();
    }

    // -------------------TOP PART------------------------------------------

    if (!truth.w  &&  !truth.x)
    {
        temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
    }

    __syncthreads();

    if (tididx > 3 && tididx <(dimens.base-4))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
	}


    k=6;
	__syncthreads();

	while(k<dimens.hts[4])
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k+=2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}
		k+=2;
		__syncthreads();
	}

    writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
    
}

__global__
void
splitDiamond(REALthree *inRight, REALthree *inLeft, REALthree *outRight, REALthree *outLeft)
{
    extern __shared__ REALthree temper[];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    int k = dimens.hts[2]; //Starts more like downTriangle

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

    //The edges are now in the center of the node 0 which is easy to find using the global id.
    const char4 truth = {gid == dimens.hts[0], gid == dimens.hts[1], gid == dimens.hts[2], gid == dimens.hts[3]};

    __syncthreads();

    //Still need to set the boundary values first because they aren't necessarily preserved in the global arrays.
    if (truth.z)
    {
        temper[tididx] = dbd[0];
        temper[tidxTop] = dbd[0];
    }
    if (truth.y)
    {
        temper[tididx] = dbd[1];
        temper[tidxTop] = dbd[1];
    }

    __syncthreads();

    while(k>0)
    {
        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();

        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();
    }

    if (!truth.y && !truth.z && threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
	}

	__syncthreads();
    k=4;

    while(k<dimens.hts[2])
    {
        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);

        }

        k+=2;
        __syncthreads();

        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }
        k+=2;
        __syncthreads();

    }

	writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}

//Now we can set the namespace.
using namespace std;

//Get energy out from conserved variables for plotting.
__host__
REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}


//Parameters are straighforward and taken directly from inputs to program.  Wrapper that clls the classic procedure.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    REALthree *dEuler_in, *dEuler_out;

    //Allocate device arrays.
    cudaMalloc((void **)&dEuler_in, sizeof(REALthree)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALthree)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    //Print to make sure we're here
    cout << "Classic scheme" << endl;

    //Start the timer (simulation timer that is.)
    double t_eq = 0.0;
    double twrite = freq - QUARTER*dt;

    //Call the kernel to step forward alternating global arrays with each call.
    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

    }

    cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dEuler_in);
    cudaFree(dEuler_out);

    return t_eq;

}

//The wrapper that enacts the swept rule.
double
sweptWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    const size_t smem = (2*dimz.base)*sizeof(REALthree); //Amt of shared memory to request

	REALthree *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

    // Allocate device global memory
	cudaMalloc((void **)&d_IC, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REALthree)*dv);
    cudaMalloc((void **)&d2_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REALthree)*dv);

    // Transfer over the initial conditions.
	cudaMemcpy(d_IC,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

	// Start the simulation time counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

    //Call up first out of loop with right and left 0
	upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

	// Call the kernels until you reach the final time

    splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
    
    double t_eq = t_fullstep;

    while(t_eq < t_end)
    {
        wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left);

        splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
        //It always ends on a left pass since the down triangle is a right pass.
        t_eq += t_fullstep;
    }
    
    // The last call is down so call it and pass the relevant data to the host with memcpy.
    downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
    cudaFree(d2_right);
	cudaFree(d2_left);
    return t_eq;
}

int main( int argc, char *argv[] )
{
    cout.precision(10); 
    
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(GPUNUM);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
	ofstream fwr;
    dimz.gam = 1.4;
    dimz.mgam = 0.4;

    bd[0].x = ONE; //Density
    bd[1].x = 0.125;
    bd[0].y = ZERO; //Velocity
    bd[1].y = ZERO;
    bd[0].z = ONE/dimz.mgam; //Energy
    bd[1].z = 0.1/dimz.mgam;

    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const double dt = atof(argv[3]); //Timestep
	const double tf = atof(argv[4]) - QUARTER*dt; //Finish time
    const double freq = atof(argv[5]); //Frequency of output (i.e. every 20 s (simulation time))
    const int scheme = atoi(argv[6]); //2 for Alternate, 1 for GPUShared, 0 for Classic
    const int bks = dv/tpb; //The number of blocks
    const double dx = lx/((REAL)dv-TWO); //Grid size.
    if (scheme) fwr.open("eulerResult/atomicS.dat", ios::trunc);
    else fwr.open("eulerResult/atomicC.dat", ios::trunc);
    fwr.precision(10);
    
    //Declare the dimensions in constant memory.
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4; // Length of the base of a node.
    dimz.idxend = dv-1; // Index of last spatial point.
    dimz.idxend_1 = dv-2; // 2nd to last spatial point.

    for (int k=-2; k<3; k++) dimz.hts[k+2] = (tpb/2) + k; //Middle values in the node (masking values)

    cout << "Euler Array --- #Blocks: " << bks << " | dt/dx: " << dimz.dt_dx << endl;

	// Initialize arrays.
    REALthree *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALthree), cudaHostAllocDefault); // Initial conditions
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALthree), cudaHostAllocDefault); // Final values

	for (int k = 0; k<dv; k++) IC[k] = (k<dv/2) ? bd[0] : bd[1]; // Populate initial conditions

    

	// Write out x length and then delta x and then delta t.
    // First item of each line is variable second is timestamp.
    #ifndef NOWRITE
	fwr << lx << " " << (dv-2) << " " << dx << " " << endl;

    fwr << "Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << "Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << "Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << endl;
    #endif

    // Transfer data to GPU in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALthree));

    // Start the counter and start the clock.
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
    cudaEventRecord( start, 0);
    string tpath;

    // Call the correct function with the correct algorithm.
    cout << scheme << " " ;
    double tfm;
    if (scheme)
    {
        tpath = "eulerResult/Array_Swept.csv";
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, IC, T_final, freq, fwr);
    }
    else
    {
        tpath = "eulerResult/Array_Classic.csv";
        tfm = classicWrapper(bks, tpb, dv, dt, tf, IC, T_final, freq, fwr);
    }

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    timed *= 1.e3;

    double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

    FILE *timeOut;
    timeOut = fopen(tpath.c_str(), "a+");    
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,nX,time\n");
    fprintf(timeOut, "%d,%d,%.8f\n", tpb, dv, per_ts);
    fclose(timeOut);

    #ifndef NOWRITE

	fwr << "Density " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << "Energy " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << energy(T_final[k]) << " ";
    fwr << endl;

    fwr << "Pressure " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_final[k]) << " ";
    fwr << endl;

    #endif

	fwr.close();

    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;
}