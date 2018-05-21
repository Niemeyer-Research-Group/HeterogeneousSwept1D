/**
	The equations specific global variables.
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

#define NSTEPS              2
#define NVARS               1
#define NSTATES             2 // How many numbers in the struct.

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
    REAL Fo;
};

#define AL 8.0e-6

//---------------//
struct states{
    REAL T[2];
    //size_t tstep; // Consider as padding.
};

typedef Json::Value jsons;
std::string fspec = "Heat";
std::string outVars[NVARS] = {"Temperature"}; //---------------//

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ eqConsts deqConsts;  //---------------//
eqConsts heqConsts; //---------------//
states bound[2];

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

#ifdef __CUDA_ARCH__
    #define DIMS    deqConsts
#else
    #define DIMS    heqConsts
#endif

__host__ double indexer(double dx, int i, int s)
{
    return dx*(i + s);
}

__host__ REAL printout(states *state, int i)
{
    return state->T[0];
}

// 12.0*rsqrt(xs+0.1) - xs*xs;
// 6.0 * sin(5.0 * M_PI * xs)
__host__ states icond(double xs)
{
    states s;
    s.T[0] = 6.0 * sin(5.0 * M_PI * xs);
    s.T[1] = s.T[0];
    return s;
}

__host__ void equationSpecificArgs(jsons inJs)
{
    REAL dtx = inJs["dt"].asDouble();
    REAL dxx = inJs["dx"].asDouble();
    heqConsts.Fo = AL*dtx/(dxx*dxx);

    if (heqConsts.Fo>.5)
    {
        std::cout << "Fourier too high: " << heqConsts.Fo << std::endl;
    }

    double lxx = inJs["lx"].asDouble();
    bound[0] = icond(0.0);
    bound[1] = icond(lxx);
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
__host__ void initialState(jsons inJs, states *inl, int idx, int xst)
{
    double dxx = inJs["dx"].asDouble();
    double xss = indexer(dxx, idx, xst);
    inl[idx] = icond(xss);
}

__host__ void mpi_type(MPI_Datatype *dtype)
{
    MPI_Type_contiguous(2, MPI_R, dtype);
    MPI_Type_commit(dtype);
}

__device__ __host__
void stepUpdate(states *heat, int idx, int tstep)
{
    int otx = tstep % NSTEPS; //Modula is output place
    int itx = (otx^1); //Opposite in input place.

    heat[idx].T[otx] = DIMS.Fo*(heat[idx-1].T[itx] + heat[idx+1].T[itx]) + (1.0-2.0*DIMS.Fo) * heat[idx].T[itx];
}
