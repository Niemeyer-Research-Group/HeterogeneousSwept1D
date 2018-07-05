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

__device__ __host__ __forceinline__
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

__host__ void mpi_type(MPI_Datatype *dtype)
{
    //double 3 type
    MPI_Datatype vtype;
    MPI_Datatype typs[] = {MPI_R, MPI_R, MPI_R};
    int n[] = {1, 1, 1};
    MPI_Aint disp[] = {0, sizeof(REAL), 2*sizeof(REAL)};

    MPI_Type_create_struct(3, n, disp, typs, &vtype);
    MPI_Type_commit(&vtype);

    int n2[] = {2, 1};
    MPI_Datatype typs2[] = {vtype, MPI_R};
    MPI_Aint disp2[] = {0, 2*sizeof(REALthree)};

    MPI_Type_create_struct(2, n2, disp2, typs2, dtype);
    MPI_Type_commit(dtype);

    MPI_Type_free(&vtype);
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
