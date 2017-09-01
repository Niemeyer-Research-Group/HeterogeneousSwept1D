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
#include <omp.h>

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
#include "json/json.h"

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

#define NSTEPS              4
#define NVARS               4

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
};

std::string outVars[4] = {"DENSITY", "VELOCITY", "ENERGY", "PRESSURE"}; //---------------// 

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ eqConsts deqConsts;  //---------------// 
eqConsts heqConsts; //---------------// 
REALthree hBounds[2]; // Boundary Conditions

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

__host__ REAL printout(const int i, states *state)
{
    REALthree subj = state->Q[0];
    REAL outs;
    switch(i)
    {
        case 0: outs = density(subj);
        case 1: outs = velocity(subj);
        case 2: outs = energy(subj);
        case 3: outs = pressure(subj);
    } 
    return outs;
}

/*
dimensions heqConsts; //---------------// 
REALthree hBound[2]; // Boundary Conditions
double lx; // Length of domain.
*/

__host__ void equationSpecificArgs(jsons inJ)
{
    heqConsts.gamma = inJ["gamma"].asDouble();
    heqConsts.mgamma = heqConsts.gamma - 1;
    REAL rhoL = inJ["rhoL"].asDouble();
    REAL vL = inJ["vL"].asDouble();
    REAL pL = inJ["pL"].asDouble();
    REAL rhoR = inJ["rhoR"].asDouble();
    REAL vR = inJ["vR"].asDouble();
    REAL pR = inJ["pR"].asDouble();
    hBounds[0].x = rhoL;
    hBounds[0].y = vL*rhoL;
    hBounds[0].z = pL/heqConsts.mgamma + HALF * rhoL * vL * vL;
    hBounds[1].x = rhoR;
    hBounds[1].y = vR*rhoR,
    hBounds[1].z = pR/heqConsts.mgamma + HALF * rhoR * vR * vR;
    REAL dtx = inJ["dt"].asDouble();
    REAL dxx = inJ["dx"].asDouble();
    heqConsts.dt_dx = dtx/dxx;
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
__host__ void initialState(jsons inJ, int idx, int xst, states *inl, double *xs)
{
    REAL dxx = inJ["dx"].asDouble();
    REAL lx = inJ["lx"].asDouble();
    double xss = dxx*(double)(idx + xst);
    xs[idx] = xss;
    bool wh = inJ["IC"].asString() == "PARTITION";
    if (wh)
    {
        int side = (xss < HALF*lx);
        inl->Q[0] = hBounds[side];
    }
}

__host__ void mpi_type(MPI_Datatype *dtype)
{ 
    //double 3 type
    MPI_Datatype vtype;
    MPI_Datatype typs[3] = {MPI_R, MPI_R, MPI_R};
    int n[3] = {1};
    MPI_Aint disp[3] = {0, sizeof(REAL), 2*sizeof(REAL)};

    MPI_Type_struct(3, n, disp, typs, &vtype);
    MPI_Type_commit(&vtype);

    typs[0] = vtype;
    typs[2] = vtype;
    disp[1] = 3*sizeof(vtype);
    disp[2] = 4*sizeof(REAL);

    MPI_Type_struct(3, n, disp, typs, dtype);
    MPI_Type_commit(dtype);

    MPI_Type_free(&vtype);
}

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
void pressureRatio(states *state, int idx, int tstep)
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
    return (QNAN(pRatio) || pRatio<0) ? qH :  (qH + HALF * QMIN(pRatio, ONE) * (qN - qH));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__ 
__forceinline__ REALthree eulerFlux(REALthree qL, REALthree qR)
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
__forceinline__ REALthree eulerSpectral(REALthree qL, REALthree qR)
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

/**
    The Final step of the finite volume scheme.

    First: The pressure ratio calculation is decomposed to avoid division and calling the limitor unnecessarily.  Although 3 pressure ratios would be required, we can see that there are only 4 unique numerators and denominators in that calculation which can be calculated without using division or calling pressure (which uses division).  The edge values aren't guaranteed to have the correct conditions so the flags set the appropriate pressure values to 0 (Pressures are equal) at the edges.
    Second:  The numerator and denominator are tested to see if the pressure ratio will be Nan or <=0. If they are, the limitor doesn't need to be called.  If they are not, call the limitor and calculate the pressure ratio.
    Third:  Use the reconstructed values at the interfaces to get the flux at the interfaces using the spectral radius and flux functions and combine the results with the flux variable.
    Fourth: Repeat for second interface and update current volume. 

    @param state  Reference to the working array in SHARED memory holding the dependent variables.
    @param idx  The indices of the stencil points.
    @param flagLeft  True if the point is the first finite volume in the tube.
    @param flagRight  True if the point is the last finite volume in the tube.
    @return  The updated value at the current spatial point.
*/
__device__ __host__ void eulerStep(states *state, int idx, int tstep)
{
    REALthree tempStateLeft, tempStateRight;

    tempStateLeft = limitor(state[idx-1].Q[tstep], state[idx].Q[tstep], state[idx-1].Pr);
    tempStateRight = limitor(state[idx].Q[tstep], state[idx-1].Q[tstep], ONE/state[idx].Pr);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Q[tstep], state[idx+1].Q[tstep], state[idx].Pr);
    tempStateRight = limitor(state[idx+1].Q[tstep], state[idx].Q[tstep], ONE/state[idx+1].Pr);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    state[idx].Q[tstep] = state[idx].Q[0] + ((QUARTER * (tstep+1)) * DIMS.dt_dx * flux);
}

__device__ __host__ 
void stepUpdate(states *state, int idx, int tstep)
{
    if (tstep & 1) //Odd 0 for even numbers
    {
        pressureRatio(state, idx, DIVMOD(tstep));
    }
    else
    {
        eulerStep(state, idx, DIVMOD(tstep));
    }
}

/*
---------------------------
    DECOMP CORE
---------------------------
*/

#define TAGS(x) x & 32767

#define CEIL(x, y)  (x + y - 1) / y 

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

// MPI process properties
MPI_Datatype struct_type;
MPI_Request req[2];
int lastproc, nprocs, ranks[3];

struct globalism {
// Topology
    int nThreads, nWaves, nGpu, nX;  
    int xg, xcpu, xWave;
    bool hasGpu;
    double gpuA;

// Geometry
    int tpb, tpbp, base, bks;
    int ht, htm, htp;
    int szState;

// Iterator
    double tf, freq, dt, dx, lx;
    bool bCond[2] = {true, true}; // Initialize passing both sides.
};

globalism cGlob;

jsons solution;
jsons timing;

//Always prepared for periodic boundary conditions.
void makeMPI(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks[1]);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
	ranks[0] = (ranks[1]-1) % nprocs;
    ranks[2] = (ranks[1]+1) % nprocs;
}

// void getDeviceInformation();
// {
//     cudaGetDeviceCount(nGpu);

//     if (nGpu)
//     {
//         cudaGetDeviceProp(&props);
//     }
    
//     nthreads = omp_get_num_procs();

//     // From this I want what GPUs each proc can see, and how many threads they can make
//     // This may require nvml to get the UUID of the GPUS, pass them all up to the 
//     // Master proc to decide which proc gets which gpu.
// }

/* 
    Takes any extra command line arguments which override json default args and inserts 
    them into the json type which will be read into variables in the next step.

    Arguments are key, value pairs all lowercase keys, no dash in front of arg.
*/
static bool mg = false;

void parseArgs(jsons inJ, int argc, char *argv[])
{
    if (argc>6)
    {
        for (int k=6; k<argc; k+=2)
        {
            inJ[argv[k]] = argv[k+1];
		// If it sets nW, flip the bool.
	    mg=true;
        }
    }
}

void initArgs(jsons inJ)
{
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.szState = sizeof(states);
    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.tpb = inJ["tpb"].asInt();
    cGlob.gpuA = inJ["gpuA"].asDouble();
    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.nX = inJ["nX"].asInt();

    // Derived quantities
    cGlob.xcpu = cGlob.nThreads * cGlob.tpb;
    cGlob.bks = (((double)cGlob.xcpu * cGlob.gpuA)/cGlob.tpb);
    cGlob.xg = cGlob.tpb * cGlob.bks;
    cGlob.gpuA = (double)cGlob.xg/(double)cGlob.tpb; // Adjusted gpuA.
    cGlob.xWave = (nprocs * cGlob.xcpu + cGlob.nGpu * cGlob.xg); 

    // Do it backward if you already know the waves. Else we get the waves from nX (which is just an approximation).
    if (mg)
    {
        cGlob.nWaves = inJ["nW"].asInt();
    }
    else
    {
        cGlob.nWaves = CEIL(cGlob.xWave, cGlob.nX);
    }

    cGlob.nX = cGlob.nWaves*cGlob.xWave;
    cGlob.tpbp = cGlob.tpb + 1;
    cGlob.base = cGlob.tpb + 2;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht - 1;
    cGlob.htp = cGlob.ht + 1;

    cGlob.dx = cGlob.lx/((double)cGlob.nX - 2.0); // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs(inJ);

    // Swept Always Passes!

    // If BCTYPE == "Dirichlet"
    if (!ranks[1]) cGlob.bCond[0] = false;
    if (ranks[1] == lastproc) cGlob.bCond[1] = false;
    // If BCTYPE == "Periodic"
        // Don't do anything.

}

void solutionOutput(states *outState, REAL tstamp, REAL xpt)
{
    for (int k=0; k<NVARS; k++)
    {
        //solution[outVars[k]][tstamp][xpt] = printout(k, outState);
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

/**
---------------------------
    CLASSIC CORE
---------------------------
*/

/**
    The Classic Functions for the stencil operation
*/

/** 
    Classic kernel for simple decomposition of spatial domain.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, ts);
}

void classicStepCPU(states *state, int numx, int tstep)
{
    if (!ranks[1]) std::cout << "we're taking a classic step on the cpu: " << tstep << std::endl;
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}

void classicPassLeft(states *state, int idxend, int tstep)
{   
    if (cGlob.bCond[0])
    {
        MPI_Isend(&state[1], 1, struct_type, ranks[0], TAGS(tstep),
                MPI_COMM_WORLD, &req[0]);

        MPI_Recv(&state[0], 1, struct_type, ranks[0], TAGS(tstep+100), 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
    if (!ranks[1]) std::cout << "we're passing a classic step Left on the cpu: " << tstep << std::endl;
}

void classicPassRight(states *state, int idxend, int tstep)
{
    if (cGlob.bCond[1]) 
    {
        MPI_Isend(&state[idxend-1], 1, struct_type, ranks[2], TAGS(tstep+100),
                MPI_COMM_WORLD, &req[1]);

        MPI_Recv(&state[idxend], 1, struct_type, ranks[2], TAGS(tstep), 
                MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    }
}

// We are working with the assumption that the parallelism is too fine to see any benefit.
// Still struggling with the idea of the local vs parameter arrays.
// Classic Discretization wrapper.
double classicWrapper(states **state, double **xpts, int *tstep)
{
    if (!ranks[1]) std::cout << "Classic Decomposition" << std::endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        const int xc = cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;
        const int cpuzise = cGlob.szState * xcpp;

        states *dState;
        
        cudaMalloc((void **)&dState, gpusize);
        // Copy the initial conditions to the device array.
        // This is ok, the whole array has been malloced.
        cudaMemcpy(dState, state[1], gpusize, cudaMemcpyHostToDevice);

        // Four streams for four transfers to and from cpu.
        cudaStream_t st1, st2, st3, st4;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);
        cudaStreamCreate(&st4);
		
	if (!ranks[1]) std::cout << "Just for fun: " << xcpp << " nums in cpu " << xgpp << " nums in GPU" << std::endl; 
        while (t_eq < cGlob.tf)
        {
            classicStep <<< cGlob.bks, cGlob.tpb >>> (dState, *tstep);

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicStepCPU(state[0], xcp, *tstep);
                }
                #pragma omp section
                {
                    classicStepCPU(state[2], xcp, *tstep);
                }
            }
            
            // Host to device first.
            # pragma omp parallel sections num_threads(3)
            {
                #pragma omp section
                {
                    cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
                    cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
                    cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
                    cudaMemcpyAsync(state[0], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4);
                }
                #pragma omp section
                {
                    classicPassRight(state[2], xcp, *tstep);
                }
                #pragma omp section
                {
                    classicPassLeft(state[0], xcp, *tstep);
                }
            }
            
            // Increment Counter and timestep
            if (MODULA(*tstep)) t_eq += cGlob.dt;
            tstep++;

            if (t_eq > twrite)
            {
                cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);                
                for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
                for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

                twrite += cGlob.freq;
            }
        }

        cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);
        
        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
        for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
        for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

        cudaFree(dState);
    }
    else
    {
        int xcp = cGlob.xcpu + 1;

        while (t_eq < cGlob.tf)
        {

            classicStepCPU(state[0], xcp, *tstep);
            if (MODULA(*tstep)) t_eq += cGlob.dt;
            *tstep++;

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicPassRight(state[0], xcp, *tstep);
                }
                #pragma omp section
                {
                    classicPassLeft(state[0], xcp, *tstep);
                }
            }

            if (t_eq > twrite)
            
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
                twrite += cGlob.freq;
            }

        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
    }
    return t_eq;
}

/*
---------------------------
    SWEPT CORE
---------------------------
*/

/**
    This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
*/

/**
    Builds an upright triangle using the swept rule.

    Upright triangle using the swept rule.  This function is called first using the initial conditions or after results are read out using downTriange.  In the latter case, it takes the result of down triangle as IC.

    @param initial Initial condition.
    @param tstep The timestep we're starting with.
*/

static int offSend[2];
static int offRecv[2];
static int cnt, turn;

void swIncrement()
{
    cnt++;
    turn = cnt & 1;
}

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
            stepUpdate(temper, tidx, tstep + k); 
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx];
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
            stepUpdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid] = temper[tidx];
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
            stepUpdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}

    #pragma unroll
	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx];
}

// HOST SWEPT ROUTINES
void upTriangleCPU(states *state, int tstep)
{
    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}


void downTriangleCPU(states *state, int tstep, int zi, int zf)
{
    for (int k=cGlob.ht; k>1; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
    for (int n=zi; n<zf; n++)
    {
        stepUpdate(state, n, tstep);
    }
}

// Now you can call this on a split for all proc/threads except (0,0)
void wholeDiamondCPU(states *state, int tstep, int zi, int zf)
{
    for (int k=cGlob.ht; k>1; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }

    for (int n=zi; n<zf; n++)
    {
        stepUpdate(state, n, tstep);
    } 

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}

void splitDiamondCPU(states *state, int tstep)
{
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }
}

static void inline passSwept(states *stateSend, states *stateRecv, int tstep)
{
    MPI_Isend(stateSend + offSend[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD, &req[turn]);

    MPI_Recv(stateRecv + offRecv[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep), 
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
}

/*
    The idea of the timestep counter is: the timestep you're on is the timestep you would export 
    if you called downTriangle NEXT.

*/

double sweptWrapper(states **state, double **xpts, int *tstep)
{
    std::cout << "Swept Decomposition" << std::endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    states **staten = new states* [cGlob.nThreads]; 
    int ar1, ptin, tid, strt; 
    const int lastThread = cGlob.nThreads-1, lastWave = cGlob.nWaves-1;
    int sw[2] = {!ranks[1], ranks[1] == lastproc}; // {First node, last node
    int stride;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        int tps = cGlob.nThreads/2; // Threads per side.
        for (int k=0; k<cGlob.nThreads; k++)
        {
            ar1 = (k/tps) * 2;
            ptin = (k % tps) * cGlob.tpb; 
            staten[k] = (state[ar1] + ptin); // ptr + offset
        }

        stride = tps * cGlob.tpb;
        const int xc =cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;

        const size_t gpusize = cGlob.szState * (xc + cGlob.htp);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;

        offSend[0] = 1, offSend[1] = xc; // {left, right}    
        offRecv[0] = xcp, offRecv[1] = 0; 

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);

        states *dState; 
        cudaMalloc((void **)&dState, gpusize);

        cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice);

        /* -- DOWN MUST FOLLOW A SPLIT AND UP CANNOT BE IN WHILE LOOP SO DO UP AND FIRST SPLIT OUTSIDE OF LOOP 
            THEN LOOP CAN BE WITH WHOLE - DIAMOND - CHECK DOWN.
        */
        // ------------ Step Forward ------------ //
        // ------------ UP ------------ //

        upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, *tstep);

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

        // ------------ Pass Edges ------------ // Give to 0, take from 2.
        // I'm doing it backward but it's the only way that really makes sense in this context.
        // Before I really abstracted this process away.  But here I have to deal with it.  
        //If you pass left first, you need to start at ht rather than 0.  That's the problem.

        passSwept(state[0], state[2], *tstep);
        cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
        cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
        swIncrement();//Increment

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState + cGlob.ht, *tstep);

        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
        {
            tid = omp_get_thread_num();
            sw[1] = (sw[1] && tid == lastThread); // Only true for last proc and last thread.
            for (int k=0; k<cGlob.nWaves; k++)
            {
                sw[1] = (sw[1] && k == lastWave);
                strt = k*stride + cGlob.ht;
                if (sw[1])
                {
                    splitDiamondCPU(staten[0] + strt, *tstep);
                }
                else
                {
                    wholeDiamondCPU(staten[tid] + strt, *tstep, 1, cGlob.tpbp);
                }
            }
        }   

        // ------------ Pass Edges ------------ //
        
        passSwept(state[2], state[0], *tstep);
        cudaMemcpyAsync(state[2] + offRecv[turn], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost, st1);
        cudaMemcpyAsync(dState, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
        swIncrement();

        // Increment Counter and timestep
        tstep += cGlob.tpb;
        t_eq += cGlob.dt * (*tstep/NSTEPS);

        while(t_eq < cGlob.tf)
        {
            // ------------ Step Forward ------------ //
            // ------------ WHOLE ------------ //
            
            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState, *tstep);

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
            {
                tid = omp_get_thread_num();
                sw[0] = (sw[0] && !tid);
                sw[1] = (sw[1] && tid == lastThread);
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    sw[0] = (sw[0] && !k);
                    sw[1] = (sw[1] && tid == lastWave);   
                    strt = k*stride;
                    wholeDiamondCPU(staten[tid] + strt, *tstep, sw[0] + 1,  cGlob.tpbp-sw[1]);
                }
            }   

            // ------------ Pass Edges ------------ //
            
            passSwept(state[0], state[2], *tstep);
            cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
            swIncrement();//Increment

            // Increment Counter and timestep
            tstep += cGlob.tpb;
            t_eq += cGlob.dt * (*tstep/NSTEPS);

            // ------------ Step Forward ------------ //
            // ------------ SPLIT ------------ //

            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState + cGlob.ht, *tstep);

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
            {
                tid = omp_get_thread_num();
                sw[1] = (sw[1] && tid == lastThread); // Only true for last proc and last thread.
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    sw[1] = (sw[1] && k == lastWave);
                    strt = k*stride + cGlob.ht;
                    if (sw[1])
                    {
                        splitDiamondCPU(staten[0] + strt, *tstep);
                    }
                    else
                    {
                        wholeDiamondCPU(staten[tid] + strt, *tstep, 1, cGlob.tpbp);
                    }
                }
            }   

            // ------------ Pass Edges ------------ //
            
            passSwept(state[2], state[0], *tstep);
            cudaMemcpyAsync(state[2] + offRecv[turn], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
            swIncrement();

            // Increment Counter and timestep
            tstep += cGlob.tpb;
            t_eq += cGlob.dt * (*tstep/NSTEPS);

            if (t_eq > twrite)
            {
                downTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState + cGlob.htp, *tstep);

                #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
                {
                    tid = omp_get_thread_num();
                    sw[0] = (sw[0] && !tid);
                    sw[1] = (sw[1] && tid == lastThread);
                    for (int k=0; k<cGlob.nWaves; k++)
                    {
                        sw[0] = (sw[0] && !k);
                        sw[1] = (sw[1] && tid == lastWave);   
                        strt = k*stride;
                        downTriangleCPU(staten[tid] + strt, *tstep, sw[0] + 1,  cGlob.tpbp-sw[1]);
                    }
                }   

                cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);
                                
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);                
                for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
                for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

                upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, *tstep);

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

                passSwept(state[0], state[2], *tstep);
                cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
                cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
                swIncrement();//Increment

                // Increment Counter and timestep
                tstep += cGlob.tpb;
                t_eq += cGlob.dt * (*tstep/NSTEPS);

                twrite += cGlob.freq;
            }
        }
        cudaFree(dState);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
    }
    // YES YOU CAN PASS THE SAME POINTER AS TWO DIFFERENT ARGS TO THE FUNCITON.
    else
    {
        int xcp = cGlob.xcpu + cGlob.htp;
        
        for (int k=0; k<cGlob.nThreads; k++)
        {
            ptin = k * cGlob.tpb; 
            staten[k] = (state[0] + ptin); // ptr + offset
        }

        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
        {
            tid = omp_get_thread_num();
            for (int k=0; k<cGlob.nWaves; k++)
            {
                strt = k*stride;
                upTriangleCPU(staten[tid] + strt, *tstep);
            }
        }

        passSwept(state[0], state[0], *tstep); // Left first
        swIncrement();

        while (t_eq < cGlob.tf)
        {

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
            {
                tid = omp_get_thread_num();
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    strt = k*stride;
                    if (!ranks[1] && !(tid + k))
                    {
                        splitDiamondCPU(staten[0], *tstep);
                    }
                    else
                    {
                        wholeDiamondCPU(staten[tid] + strt, *tstep, 1, cGlob.tpbp);
                    }
                }
            }   

            passSwept(state[0], state[0], *tstep); // Left first
            swIncrement();

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

/**
----------------------
    MAIN PART
----------------------
*/

#ifndef HDW
    #define HDW     "src/hardware/WORKSTATION.json"
#endif

std::vector<int> jsonP(jsons jp, size_t sz)
{
	std::vector <int> outv;
	for(int i=0; i<sz; i++)
	{
		outv.push_back(jp[i].asInt());
	}
	return outv;
}

int main(int argc, char *argv[])
{   
    makeMPI(argc, argv);

    std::string ext = ".json";
    std::string myrank = std::to_string(ranks[1]);
    std::string sout = argv[3];
    sout.append(myrank);
    sout.append(ext); 
    std::string scheme = argv[1];

    std::ifstream hwjson(HDW, std::ifstream::in);
    jsons hwJ;
    hwjson >> hwJ;
    hwjson.close();

    std::vector<int> gpuvec = jsonP(hwJ["GPU"], 1);
    std::vector<int> smGpu(gpuvec.size());
    std::vector<int> threadv =  jsonP(hwJ["nThreads"], 1);
    cGlob.nThreads=threadv[ranks[1]]; // Potetntial for non constant
    cGlob.hasGpu = gpuvec[ranks[1]];
    std::partial_sum(gpuvec.begin(), gpuvec.end(), smGpu.begin());
    cGlob.nGpu = smGpu.back();
    smGpu.insert(smGpu.begin(), 0);
    std::vector <int> myGPU = jsonP(hwJ["gpuID"], 1);
    int gpuID = myGPU[ranks[1]];
    
    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    jsons inJ;
    injson >> inJ;
    injson.close();

    parseArgs(inJ, argc, argv);
    initArgs(inJ);

    /*  
        Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */

    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * cGlob.hasGpu * smGpu[ranks[1]]; //
    states **state;
    double **xpts;

    int exSpace = (!scheme.compare("S")) ? cGlob.htp : 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int xalloc = xc + exSpace;

    if (cGlob.hasGpu)
    {
        cudaSetDevice(gpuID);
        
        state = new states* [3];
        xpts = new double* [3];
        cudaHostAlloc((void **) &xpts[0], xc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &xpts[1], (cGlob.xg + exSpace) * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &xpts[2], xc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[1], (cGlob.xg + exSpace) * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[2], xalloc * cGlob.szState, cudaHostAllocDefault);

        int pone = (strt + xc);
        int ptwo = (pone + cGlob.xg);

        for (int k=1; k <= xc; k++) 
        {
            initialState(inJ, k, strt, state[0], xpts[0]); 
            initialState(inJ, k, ptwo, state[2], xpts[2]); 
        }

        for (int k=1; k <= cGlob.xg; k++)  initialState(inJ, k, pone, state[1], xpts[1]); 

        cudaMemcpyToSymbol(&deqConsts, &heqConsts, sizeof(eqConsts));

        if (sizeof(REAL)>6) 
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
    }
    else 
    {
        state = new states* [1];
        xpts = new double* [1];
        cudaHostAlloc((void **) &xpts[0], xalloc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        for (int k=1; k<=xc; k++)  initialState(inJ, k, strt, state[0], xpts[0]); 
    }

    int tstep = 1;
    // Start the counter and start the clock.
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    // Call the correct function with the correct algorithm.
    double tfm;

    if (!scheme.compare("C"))
    {
        tfm = classicWrapper(state, xpts, &tstep);
    }
    else if  (!scheme.compare("S"))
    {
//        tfm = sweptWrapper(state, xpts, &tstep);
    }
    else
    {
        std::cerr << "Incorrect or no scheme given" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timed, start, stop);

    endMPI();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    std::ofstream soljson(argv[3]);
    soljson << solution;
    soljson.close();

    if (!ranks[1])
    {
        //READ OUT JSONS
        
        timed *= 1.e3;

        double n_timesteps = tfm/cGlob.dt;

        double per_ts = timed/n_timesteps;

        std::cout << n_timesteps << " timesteps" << std::endl;
        std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

        jsons timing;
        //timing[cGlob.nX][cGlob.tpb][cGlob.gpuA] = per_ts;

        std::ofstream timejson(argv[4]);
        //timejson << timing;
        timejson.close();
    }

    if (cGlob.hasGpu)
    {
        cudaDeviceSynchronize();
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        for (int k=0; k<3; k++)
        {
            cudaFreeHost(xpts[k]);
            cudaFreeHost(state[k]);
        }
        
        delete[] xpts;
        delete[] state;
        cudaDeviceReset();
    }
    else
    {
        cudaFreeHost(xpts[0]);
        cudaFreeHost(state[0]);
        delete[] xpts;
        delete[] state;
    }
	return 0;
}
