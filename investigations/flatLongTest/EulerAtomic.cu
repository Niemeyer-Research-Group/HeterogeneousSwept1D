/**
    Lenthening strategy for Euler - Atomic stages.
    
    COMPILE:
    nvcc EulerAtomic.cu -o ./bin/EulerAtomic -gencode arch=compute_35,code=sm_35 -lm -restrict -Xptxas=-v 
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

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

using namespace std;

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

// Hardwire in the length of the 
const double lx = 1.0;
const double GAMMA = 1.4;
const double MGAMMA = 0.4;

int nX, tpb, tpbp, base, ht, htm, htp, bks, scheme;
size_t szState, gridmem;
double tf, dt, dx, freq;

__host__ __device__
__forceinline__
int mod(int a, int b) { return (a % b + b) % b; }

//---------------//
struct eqConsts {
    REAL gamma; // Heat capacity ratio
    REAL mgamma; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int nX, nXp;

    __host__ __device__
    int modo(int idx) {return mod(idx, nX); }
};

//---------------//
struct states {
    REALthree Q[2]; // Full Step, Midpoint step state variables
    REAL Pr; // Pressure ratio
};

std::string outVars[NVARS] = {"DENSITY", "VELOCITY", "ENERGY", "PRESSURE"}; 

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/

__constant__ eqConsts deqConsts;  //---------------//
eqConsts heqConsts;               //---------------//

REALthree hBounds[2] = {{1.0, 0.0, 2.5}, {0.125, 0.0, 0.25}}; 
__constant__ REALthree dBounds[2];
states bound[2];

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

__device__ __host__ __forceinline__
REAL pressure(REALthree qH)
{
    return DIMS.mgamma * (qH.z - (HALF * qH.y * qH.y/qH.x));
}


__device__ __forceinline__
REAL pressureRoe(REALthree qH)
{
    return DIMS.mgamma * (qH.z - HALF * qH.y * qH.y);
}


__device__ __host__ __forceinline__
void pressureRatio(states *state, const int idx, const int tstep)
{
    state[idx].Pr = (pressure(state[idx+1].Q[tstep]) - pressure(state[idx].Q[tstep]))/(pressure(state[idx].Q[tstep]) - pressure(state[idx-1].Q[tstep]));
}


__device__ __forceinline__
REALthree limitor(REALthree qH, REALthree qN, REAL pRatio)
{
    return (QNAN(pRatio) || (pRatio<1.0e-8)) ? qH : (qH + HALF * QMIN(pRatio, ONE) * (qN - qH));
}


__device__ __forceinline__
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

__device__ __forceinline__
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

__device__ 
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

__device__ 
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

__global__ 
void classicStep(states *state, const int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)
    if (gid<DIMS.nXp) stepUpdate(state, gid, ts);
}

__global__ 
void upTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;
    int tnow = tstep;
	sharedstate[tidx] = state[gid];

    __syncthreads();

	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepUpdate(sharedstate, tidx, tnow);
        }
        tnow++;
		__syncthreads();
	}
    state[gid] = sharedstate[tidx];
}

__global__ 
void downTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid;
    int tidx = tid + 1;

    int tnow = tstep; // read tstep into register.

    if (tid<2 && gid>0) sharedstate[tid] = state[gid-1];
    __syncthreads();
    if ((gid + 1) < DIMS.nX) sharedstate[tid+2] = state[gid+1];
    __syncthreads();

    if (gid == 0 || gid == (DIMS.nXp)) return;

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

__global__ 
void wholeDiamond(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

    int k;
    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid;
    int tidx = tid + 1;

    int tnow = tstep; // read tstep into register.

    if (tid<2 && gid>0) sharedstate[tid] = state[gid-1];
    __syncthreads();
    if ((gid + 1) < DIMS.nX) sharedstate[tid+2] = state[gid+1];
    __syncthreads();

    if (gid == 0 || gid == (DIMS.nXp)) return;

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
    state[gid] = sharedstate[tidx];
}

__global__ 
void splitDiamond(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

    int k;
    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid + mid;
    int tidx = tid + 1;
    int gidx = DIMS.modo(gid);

    int tnow = tstep; // read tstep into register;

    if (tid<2) sharedstate[tid] = state[DIMS.modo(gid - 1)];
    __syncthreads();
    sharedstate[tid+2] = state[DIMS.modo(gid + 1)];
    __syncthreads();
    
    if (gid == DIMS.nX || gid == DIMS.nXp) return;

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
    state[gidx] = sharedstate[tidx];
}

__host__ inline 
REAL printout(states *state, int i)
{
    REALthree subj = state->Q[0];
    REAL ret;

    if (i == 0) ret = density(subj);
    if (i == 1) ret = velocity(subj);
    if (i == 2) ret = energy(subj);
    if (i == 3) ret = pressure(subj);

    return ret;
}

ofstream fwr;

void writeOut(states *outState, double tstamp)
{
    for (int i = 0; i<4; i++)
    {   
        fwr << outVars[i] << " " << tstamp << " ";
        for (int k = 1; k<(nX-1); k++)
        {
            fwr << printout(outState + k, i) << " ";
        }
        fwr << endl;
    }
}

// Classic Discretization wrapper.
double classicWrapper(states *state, int *tstep)
{
    int k, tmine = *tstep;
    double t_eq = 0.0;

    cout << "Classic Decomposition ---- " << tpb << " - " << nX << endl;

    states *dState;
    cudaMalloc((void **)&dState, gridmem);
    cudaMemcpy(dState, state, gridmem, cudaMemcpyHostToDevice);

    while (t_eq < tf)
    {
        #pragma unroll
        for (k=0; k<NSTEPS; k++)
        {
            classicStep<<<bks, tpb>>> (dState, tmine + k);
        }

        // Increment Counter and timestep
        t_eq += dt;
        tmine += NSTEPS;
    }

    cudaMemcpy(state, dState, gridmem, cudaMemcpyDeviceToHost);

    cudaFree(dState);
    return t_eq;
}

double sweptWrapper(states *state, int *tstep)
{
    int k, tmine = *tstep;
    double t_eq = 0.0;
	cout << "Swept Decomposition ---- " << tpb << " - " << nX << endl;
    const size_t smem = (2+tpb) * szState; 
    const double multiplier = dt/((double)NSTEPS); 
    int rndme;

    states *dState;

    cudaMalloc((void **)&dState, gridmem);
    cudaMemcpy(dState, state, gridmem, cudaMemcpyHostToDevice);

    // ------------ Step Forward ------------ //
    // ------------ UP ------------ //

    upTriangle <<<bks, tpb, smem>>> (dState, tmine);
    
    // ------------ Step Forward ------------ //
    // ------------ SPLIT ------------ //

    splitDiamond <<<bks, tpb, smem>>> (dState, tmine);

    // Increment Counter and timestep
    tmine += ht;
    t_eq = tmine * multiplier;


    while(t_eq < tf)
    {
        // ------------ Step Forward ------------ //
        // ------------ WHOLE ------------ //

        wholeDiamond <<<bks, tpb, smem>>> (dState, tmine);
       

        // Increment Counter and timestep
        tmine += ht;
        t_eq = tmine * multiplier;

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        splitDiamond <<<bks, tpb, smem>>> (dState, tmine);

        tmine += ht;
        t_eq = tmine * multiplier;
    }

    downTriangle <<<bks, tpb, smem>>> (dState, tmine);

    tmine += ht;
    rndme = tmine/NSTEPS;
    t_eq = rndme * dt;

    cudaMemcpy(state, dState, gridmem, cudaMemcpyDeviceToHost);

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

    cout.precision(10); 

	cudaSetDevice(GPUNUM);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    nX = atoi(argv[1]); //Number of spatial points
    tpb = atoi(argv[2]); //Threads per Block
    szState = sizeof(states);
    dt = atof(argv[3]); //Timestep
	tf = atof(argv[4]) - QUARTER*dt; //Finish time
    freq = atof(argv[5]); //Frequency of output (i.e. every 20 s (simulation time))
    scheme = atoi(argv[6]); 
    dx = lx/((REAL)nX-TWO); //Grid size.
    bks = nX/tpb; //The number of blocks
    gridmem = szState * nX;
    ht = tpb/2;

    if (scheme) fwr.open("eulerResult/atomicS.dat", ios::trunc);
    else fwr.open("eulerResult/atomicC.dat", ios::trunc);
    fwr.precision(10);

    heqConsts.gamma = GAMMA;
    heqConsts.mgamma = MGAMMA;
    heqConsts.dt_dx = dt/dx;
    heqConsts.nX = nX;
    heqConsts.nXp = nX-1;

    cout << "Euler Atomic --- #Blocks: " << bks << " | dt/dx: " << heqConsts.dt_dx << endl;

    states *state;
    cudaHostAlloc((void **) &state, gridmem, cudaHostAllocDefault);
    REALthree filler;

    for (int k = 0; k<nX; k++) 
    {
        filler = (k<nX/2) ? hBounds[0] : hBounds[1]; 
        state[k].Q[0] = filler;
        state[k].Q[1] = filler;
        state[k].Pr = 0;
    }

    cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));
    //cudaMemcpyToSymbol(dBounds, &hBounds, 2*szState);

    #ifndef NOWRITE
        writeOut(state, 0.0);
    #endif

    int tstep = 1;

    // Start the counter and start the clock.
    cudaEvent_t start, stop;
    float timed;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0);

    cout << scheme << " " ;
    double tfm;
    std::string tpath;

    if (scheme)
    {
        tpath = "eulerResult/Atomic_Swept.csv";
        tfm = sweptWrapper(state, &tstep);
    }
    else
    {
        tpath = "eulerResult/Atomic_Classic.csv";
        tfm = classicWrapper(state, &tstep);
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timed, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    #ifndef NOWRITE
        writeOut(state, tfm);
    #endif
    fwr.close();

    timed *= 1.e3;
    double n_timesteps = tfm/dt;
    double per_ts = timed/n_timesteps;

    std::cout << n_timesteps << " timesteps" << std::endl;
    std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

    FILE *timeOut;
    timeOut = fopen(tpath.c_str(), "a+");
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,nX,time\n");
    fprintf(timeOut, "%d,%d,%.8f\n", tpb, nX, per_ts);
    fclose(timeOut);

    cudaDeviceSynchronize();
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFreeHost(state);
    cudaDeviceReset();

    return 0;
}