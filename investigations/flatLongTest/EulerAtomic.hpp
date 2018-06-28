/**
    The equation specific functions.
*/

/**
	The equations specific global variables and function prototypes.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <mpi.h>

#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "myVectorTypes.h"
#include "json/json.h"\

using namespace std;

// We're just going to assume doubles
#define REAL            double
#define REALtwo         double2
#define REALthree       double3
#define MPI_R           MPI_DOUBLE
#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0
#define SQUAREROOT(x)   sqrt(x)

#define NSTEPS              4 // How many substeps in timestep.
#define NVARS               4 // How many variables to output.
#define NSTATES             7 // How many numbers in the struct.

// Since anyone would need to write a header and functions file, why not just hardwire this.
// If the user's number of steps isn't a power of 2 use the other one.

#define MODULA(x)           x & (NSTEPS-1)
// #define MODULA(x)           x % NSTEPS

#define DIVMOD(x)           (MODULA(x)) >> 1
/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

//---------------//
struct eqConsts {
    REAL gamma; // Heat capacity ratio
    REAL mgamma; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
};

//---------------//
struct states {
    REALthree Q[2]; // Full Step, Midpoint step state variables
    REAL Pr; // Pressure ratio
    // size_t tstep; // Consider this for padding.  Unfortunately, requires much refactoring.
};

std::string outVars[NVARS] = {"DENSITY", "VELOCITY", "ENERGY", "PRESSURE"}; //---------------//
std::string fspec = "Euler";

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ eqConsts deqConsts;  //---------------//
eqConsts heqConsts; //---------------//
REALthree hBounds[2]; // Boundary Conditions
states bound[2];

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

typedef Json::Value jsons;

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.

    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
#ifdef __CUDA_ARCH__
    #define DIMS    deqConsts
    #define QNAN(x) isnan(x)
    #define QMIN(x, y) min(x, y)
#else
    #define DIMS    heqConsts
    #define QNAN(x) std::isnan(x)
    #define QMIN(x, y) std::min(x, y)
#endif

__host__ double indexer(double dx, int i, int x)
{
    double pt = i+x;
    double dx2 = dx/2.0;
    return dx*pt - dx2;
}

__host__ REAL density(REALthree subj)
{
    return subj.x;
}

__host__ REAL velocity(REALthree subj)
{
    return subj.y/subj.x;
}

__host__ REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

__device__ __host__
__forceinline__
REAL pressure(REALthree qH)
{
    return DIMS.mgamma * (qH.z - (HALF * qH.y * qH.y/qH.x));
}

__host__ inline REAL printout(states *state, int i)
{
    REALthree subj = state->Q[0];
    REAL ret;

    if (i == 0) ret = density(subj);
    if (i == 1) ret = velocity(subj);
    if (i == 2) ret = energy(subj);
    if (i == 3) ret = pressure(subj);

    return ret;
}


__host__ inline states icond(double xs, double lx)
{
    states s;
    int side = (xs > HALF*lx);
    s.Q[0] = hBounds[side];
    s.Q[1] = hBounds[side];
    s.Pr = 0.0;
    return s;
}

__host__ void equationSpecificArgs(jsons inJs)
{
    heqConsts.gamma = inJs["gamma"].asDouble();
    heqConsts.mgamma = heqConsts.gamma - 1;
    double lx = inJs["lx"].asDouble();
    REAL rhoL = inJs["rhoL"].asDouble();
    REAL vL = inJs["vL"].asDouble();
    REAL pL = inJs["pL"].asDouble();
    REAL rhoR = inJs["rhoR"].asDouble();
    REAL vR = inJs["vR"].asDouble();
    REAL pR = inJs["pR"].asDouble();
    hBounds[0].x = rhoL;
    hBounds[0].y = vL*rhoL;
    hBounds[0].z = pL/heqConsts.mgamma + HALF * rhoL * vL * vL;
    hBounds[1].x = rhoR;
    hBounds[1].y = vR*rhoR,
    hBounds[1].z = pR/heqConsts.mgamma + HALF * rhoR * vR * vR;
    REAL dtx = inJs["dt"].asDouble();
    REAL dxx = inJs["dx"].asDouble();
    heqConsts.dt_dx = dtx/dxx;
    bound[0] = icond(0.0, lx);
    bound[1] = icond(lx, lx);
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
__host__ void initialState(jsons inJs, states *inl, int idx, int xst)
{
    double dxx = inJs["dx"].asDouble();
    double xss = indexer(dxx, idx, xst);
    double lx = inJs["lx"].asDouble();
    bool wh = inJs["IC"].asString() == "PARTITION";
    if (wh)
    {
        inl[idx] = icond(xss, lx);
    }
}


/*
    // MARK : Equation procedure
*/
__device__ __host__
__forceinline__
REAL pressureRoe(REALthree qH)
{
    return DIMS.mgamma * (qH.z - HALF * qH.y * qH.y);
}

/**
    Ratio
*/
__device__ __host__
__forceinline__
void pressureRatio(states *state, const int idx, const int tstep)
{
    state[idx].Pr = (pressure(state[idx+1].Q[tstep]) - pressure(state[idx].Q[tstep]))/(pressure(state[idx].Q[tstep]) - pressure(state[idx-1].Q[tstep]));
}

/**
    Reconstructs the state variables if the pressure ratio is finite and positive.

    @param cvCurrent  The state variables at the point in question.
    @param cvOther  The neighboring spatial point state variables.
    @param pRatio  The pressure ratio Pr-Pc/(Pc-Pl).
    @return The reconstructed value at the current side of the interface.
*/
__device__ __host__
__forceinline__
REALthree limitor(REALthree qH, REALthree qN, REAL pRatio)
{
    return (QNAN(pRatio) || (pRatio<1.0e-8)) ? qH : (qH + HALF * QMIN(pRatio, ONE) * (qN - qH));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__
__forceinline__
REALthree eulerFlux(const REALthree qL, const REALthree qR)
{
    REAL uLeft = qL.y/qL.x;
    REAL uRight = qR.y/qR.x;

    REAL pL = pressure(qL);
    REAL pR = pressure(qR);

    REALthree flux;
    flux.x = (qL.y + qR.y);
    flux.y = (qL.y*uLeft + qR.y*uRight + pL + pR);
    flux.z = (qL.z*uLeft + qR.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}

/**
    Finds the spectral radius and applies it to the interface.

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The spectral radius multiplied by the difference of the reconstructed values
*/
__device__ __host__
__forceinline__
REALthree eulerSpectral(const REALthree qL, const REALthree qR)
{
    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(qL.x);
    REAL rhoRightsqrt = SQUAREROOT(qR.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*qR.y + rhoRightsqrt*qL.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*qR.z + rhoRightsqrt*qL.z)*halfDenom;

    REAL pH = pressureRoe(halfState);

    return (SQUAREROOT(pH * DIMS.gamma) + fabs(halfState.y)) * (qL - qR);
}

__device__ __host__
void eulerStep(states *state, const int idx, const int tstep)
{
    REALthree tempStateLeft, tempStateRight;
    const int itx = (tstep ^ 1);

    tempStateLeft = limitor(state[idx-1].Q[itx], state[idx].Q[itx], state[idx-1].Pr);
    tempStateRight = limitor(state[idx].Q[itx], state[idx-1].Q[itx], ONE/state[idx].Pr);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Q[itx], state[idx+1].Q[itx], state[idx].Pr);
    tempStateRight = limitor(state[idx+1].Q[itx], state[idx].Q[itx], ONE/state[idx+1].Pr);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    state[idx].Q[tstep] = state[idx].Q[0] + ((QUARTER * (itx+1)) * DIMS.dt_dx * flux);
}

__device__ __host__
void stepUpdate(states *state, const int idx, const int tstep)
{
   int ts = DIVMOD(tstep);
    if (tstep & 1) //Odd - Rslt is 0 for even numbers
    {
        pressureRatio(state, idx, ts);
    }
    else
    {
        eulerStep(state, idx, ts);
    }
}

__global__ void classicStep(states *state, const int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)
    stepUpdate(state, gid, ts);
}
typedef std::vector<int> ivec;

states ssLeft[3];
states ssRight[3];

__global__ void upTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;

    // Using tidx as tid is kind of confusing for reader but looks valid.

	sharedstate[tidx] = state[gid + 1];

    __syncthreads();

	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepUpdate(sharedstate, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid + 1] = sharedstate[tidx];
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
*/
__global__
void
downTriangle(states *state, const int tstep, const int offset)
{
	extern __shared__ states sharedstate[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid + offset;
    int tidx = tid + 1;

    int tnow = tstep; // read tstep into register.

    if (tid<2) sharedstate[tid] = state[gid];
    __syncthreads();
    sharedstate[tid+2] = state[gid+2];
    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
	    	stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}
    state[gid] = sharedstate[tidx];
}

// __global__
// void
// printgpu(states *state, int n, int on)
// {
//     printf("%i\n", on);
//     for (int k=0; k<n; k++) printf("%i %.2f\n", k, state[k].T[on]);
// }

/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param state The working array of structures states.
    @param tstep The count of the first timestep.
*/
__global__ void
wholeDiamond(states *state, const int tstep, const int offset)
{
	extern __shared__ states sharedstate[];

    int tidx = threadIdx.x + 1; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + threadIdx.x + offset;

    int tnow = tstep;
	int k;
    if (threadIdx.x<2) sharedstate[threadIdx.x] = state[gid];
    __syncthreads();
    sharedstate[tidx+1] = state[gid + 2];

    __syncthreads();

	for (k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
        	stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}

	for (k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
    }
    state[gid + 1] = sharedstate[tidx];
}

__global__ void splitDiamondCPU(states *state, int tnow)
{
    ssLeft[2] = bound[1];
    ssRight[0] = bound[0];
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n == cGlob.ht)
            {
                ssLeft[0] = state[n-1], ssLeft[1] = state[n];
                stepUpdate(&ssLeft[0], 1, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], 1, tnow);
                state[n] = ssRight[1];
            }
            else
            {
                stepUpdate(state, n, tnow);
            }
        }
        tnow++;
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n == cGlob.ht)
            {
                ssLeft[0] = state[n-1], ssLeft[1] = state[n];
                stepUpdate(&ssLeft[0], 1, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], 1, tnow);
                state[n] = ssRight[1];
            }
            else
            {
                stepUpdate(state, n, tnow);
            }
        }
	tnow++;
    }
}

// Classic Discretization wrapper.
double classicWrapper(states **state, int *tstep)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    states putSt[2];
    states getSt[2];
    int t0, t1;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        cout << "Classic Decomposition GPU" << endl;
        const int xc = cGlob.xcpu/2;
        int xcp = xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;

        states *dState;

        cudaMalloc((void **)&dState, gpusize);
        // Copy the initial conditions to the device array.
        // This is ok, the whole array has been malloced.
        cudaMemcpy(dState, state[1], gpusize, cudaMemcpyHostToDevice);

        cout << "Entering Loop" << endl;

        while (t_eq < cGlob.tf)
        {
            //TIMEIN;
            classicStep<<<cGlob.gBks, cGlob.tpb>>> (dState, tmine);

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                writeOut(state, t_eq);
                twrite += cGlob.freq;
            }
        }

        cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

        cudaFree(dState);
    }


    return t_eq;
}

double sweptWrapper(states **state,  int *tstep)
{
	if (!ranks[1]) cout << "SWEPT Decomposition " << cGlob.tpb << endl;
	FILE *diagDump;
	std::string fname = "edge/edgeWrite_" + std::to_string(ranks[1]) + ".csv";
	diagDump = fopen(fname.c_str(), "w+");
    const int bkL = cGlob.cBks - 1;
    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    int tmine = *tstep;

    // Must be declared global in equation specific header.

    int tou = 2000;

    cout << ranks[1] << " hasGpu" << endl;
    const int xc = cGlob.xcpu/2, xcp=xc+1;
    const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
    const int cmid = cGlob.cBks/2;
    int bx, ix;

    const size_t gpusize = cGlob.szState * (xgpp + cGlob.ht);
    const size_t ptsize = cGlob.szState * xgpp;
    const size_t passsize =  cGlob.szState * cGlob.htp;
    const size_t smem = cGlob.szState * cGlob.base;

    int gpupts = gpusize/cGlob.szState;

    cudaStream_t st1, st2;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);

    states *dState;

    cudaMalloc((void **)&dState, gpusize);
    cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice);

    /* RULES
    -- DOWN MUST FOLLOW A SPLIT 
    -- UP CANNOT BE IN WHILE LOOP 
    -||- ACTION: DO UP AND FIRST SPLIT OUTSIDE OF LOOP 
    -||- THEN LOOP CAN BEGIN WITH WHOLE DIAMOND.
    */

    // ------------ Step Forward ------------ //
    // ------------ UP ------------ //

    upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

    for (int k=0; k<cGlob.cBks; k++)
    {
        bx = (k/cmid);
        ix = 2*bx;
        upTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
    }

    // ------------ Pass Edges ------------ //
    // -- FRONT TO BACK -- //

    cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

    passSwept(state[0] + 1, state[2] + xcp, tmine, 0);

    // ------------ Step Forward ------------ //
    // ------------ SPLIT ------------ //

    wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

    for (int k=0; k<(cGlob.cBks); k++)
    {
        bx = (k/cmid);
        ix = 2*bx;
        if ((ranks[1] == lastproc) && (k == bkL))
        {
            splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
        }
        else
        {
            wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
        }
    }

    // ------------ Pass Edges ------------ //
    // -- BACK TO FRONT -- //

    cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

    passSwept(state[2] + xc, state[0], tmine+1, 1);

    // Increment Counter and timestep
    tmine += cGlob.ht;
    t_eq = cGlob.dt * (tmine/NSTEPS);

    if (!ranks[1]) state[0][0] = bound[0];
    if (ranks[1]==lastproc) state[2][xcp] = bound[1];

    while(t_eq < cGlob.tf)
    {
        // ------------ Step Forward ------------ //
        // ------------ WHOLE ------------ //

        wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

        for (int k=0; k<cGlob.cBks; k++)
        {
            bx = (k/cmid);
            ix = 2*(k/cmid);
            wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
        }

        // ------------ Pass Edges ------------ //

        cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

        passSwept(state[0] + 1, state[2] + xcp, tmine, 0);


        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

        for (int k=0; k<(cGlob.cBks); k++)
        {
            bx = (k/cmid);
            ix = 2*bx;
            if ((ranks[1] == lastproc) && (k == bkL))
            {
                splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
            }
            else
            {
                wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
            }
        }

        // ------------ Pass Edges ------------ //
        // -- BACK TO FRONT -- //
        cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

        passSwept(state[2] + xc, state[0], tmine, 1);

        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);

        if (!ranks[1]) state[0][0] = bound[0];
        if (ranks[1]==lastproc) state[2][xcp] = bound[1];

        if (t_eq > twrite)
        {
            downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

            for (int k=0; k<cGlob.cBks; k++)
            {
                bx = (k/cmid);
                ix = 2*bx;
                downTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
            }

            // Increment Counter and timestep
            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);

            cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);

            writeOut(state, t_eq);
            
            upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

            for (int k=0; k<cGlob.cBks; k++)
            {
                bx = (k/cmid);
                ix = 2*bx;
                upTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
            }

            cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
            cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

            passSwept(state[0] + 1, state[2] + xcp, tmine, 0);


            // ------------ Step Forward ------------ //
            // ------------ SPLIT ------------ //

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

            for (int k=0; k<(cGlob.cBks); k++)
            {
                bx = (k/cmid);
                ix = 2*bx;
                if ((ranks[1] == lastproc) && (k ==bkL))
                {
                    splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                }
            }

            // ------------ Pass Edges ------------ //
            // -- BACK TO FRONT -- //
            cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
            cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

            passSwept(state[2] + xc, state[0], tmine+1, 1);

            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);
            twrite += cGlob.freq;
            if (!ranks[1]) state[0][0] = bound[0];
            if (ranks[1]==lastproc) state[2][xcp] = bound[1];
        }

    }

    downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

    for (int k=0; k<cGlob.cBks; k++)
    {
        bx = (k/cmid);
        ix = 2*bx;
        downTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
    }

    // Increment Counter and timestep
    tmine += cGlob.ht;
    t_eq = cGlob.dt * (tmine/NSTEPS);

    cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);

    cudaFree(dState);

    *tstep = tmine;
    return t_eq;
}


/**
----------------------
    MAIN PART
----------------------
*/

int main(int argc, char *argv[])
{

    std::string i_ext = ".json";
    std::string t_ext = ".csv";
    std::string myrank = std::to_string(ranks[1]);
    std::string scheme = argv[1];

    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();

    int prevGpu=0; //Get the number of GPUs in front of the current process.
    int gpuPlaces[nprocs]; //Array of 1 or 0 for number of GPUs assigned to process

    cGlob.xStart = cGlob.xcpu * ranks[1] + cGlob.xg * prevGpu;
    states **state;

    int exSpace = ((int)!scheme.compare("S") * cGlob.ht) + 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int nrows = (cGlob.hasGpu) ? 3 : 1;
    int xalloc = xc + exSpace;

    std::string pth = string(argv[3]);


    if (cGlob.hasGpu)
    {
        state = new states* [3];
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[1], (cGlob.xg + exSpace) * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[2], xalloc * cGlob.szState, cudaHostAllocDefault);

        cout << "Rank: " << ranks[1] << " has a GPU" << endl;
        int ii[3] = {xc, cGlob.xg, xc};
        int xi;
        for (int i=0; i<3; i++)
        {
            xi = cGlob.xStart-1;
            for (int n=0; n<i; n++) xi += ii[n];
            for (int k=0; k<(ii[i]+2); k++)  initialState(inJ, state[i], k, xi);
        }

        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

//        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
    else
    {
        state = new states*[1];
        state[0] = new states[xalloc * cGlob.szState];
        for (int k=0; k<(xc+2); k++)  initialState(inJ, state[0], k, cGlob.xStart-1);
    }

    writeOut(state, 0.0);


    if (!scheme.compare("C"))
    {
        tfm = classicWrapper(state, &tstep);
    }
    else if  (!scheme.compare("S"))
    {
        tfm = sweptWrapper(state, &tstep);
    }
    else
    {
        std::cerr << "Incorrect or no scheme given" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (!ranks[1]) timed = (MPI_Wtime() - timed);

    if (cGlob.hasGpu)  
    {
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        cudaDeviceSynchronize();
    }

    writeOut(state, tfm);

    if (!ranks[1])
    {
        timed *= 1.e6;

        double n_timesteps = tfm/cGlob.dt;

        double per_ts = timed/n_timesteps;

        std::cout << n_timesteps << " timesteps" << std::endl;
        std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

        // Write out performance data as csv
        std::string tpath = pth + "/t" + fspec + scheme + t_ext;
        FILE * timeOut;
        timeOut = fopen(tpath.c_str(), "a+");
        fseek(timeOut, 0, SEEK_END);
        int ft = ftell(timeOut);
        if (!ft) fprintf(timeOut, "tpb,gpuA,nX,time\n");
        fprintf(timeOut, "%d,%.4f,%d,%.8f\n", cGlob.tpb, cGlob.gpuA, cGlob.nX, per_ts);
        fclose(timeOut);
    }
    
        //WRITE OUT JSON solution to differential equation

	#ifndef NOS
        std::string spath = pth + "/s" + fspec + "_" + myrank + i_ext;
        std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
        if (!ranks[1]) solution["meta"] = inJ;
        soljson << solution;
        soljson.close();
	#endif

    if (cGlob.hasGpu)
    {
        for (int k=0; k<3; k++) cudaFreeHost(state[k]);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    else
    {
        delete[] state[0];
    }
    delete[] state;

    return 0;
}