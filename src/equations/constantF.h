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
    REAL n;
};

//---------------// 
struct states{
    REAL u[2];
};

typedef Json::Value jsons;
std::string fspec = "Const";
std::string outVars[NVARS] = {"NonDim"}; //---------------// 

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ eqConsts deqConsts;  //---------------// 
eqConsts heqConsts; //---------------// 
states bound[2];
int stPass, numPass; // Number of Passing states, total numbers in passing states.
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
    return state->u[0];
}

__host__ inline void unstructify(states *putSt, REAL *putReal)
{
    int gap;
    for (int k=0; k<numPass; k+=NSTATES)
    {
        gap = k*NSTATES;
        putReal[gap] = putSt[k].u[0];
        putReal[gap+1] = putSt[k].u[1];
    }
}

// Make the struct an array a struct.
__host__ inline void restructify(states *getSt, REAL *getReal)
{
    int gap;
    for (int k=0; k<numPass; k+=NSTATES)
    {
        gap = k*NSTATES;
        getSt[k].u[0] = getReal[gap];
        getSt[k].u[1]= getReal[gap+1];
    }
}


__host__ states icond(double ix, double xs)
{
    states s;
    s.u[0] = (ix*2.0)/xs;
    s.u[1] = s.u[0];
    return s;
}

__host__ void equationSpecificArgs(jsons inJs)
{
    heqConsts.n = 1.0;
}

__host__ void initialState(jsons inJs, states *inl, int idx, int xst)
{
    double dxx = inJs["dx"].asDouble();
    double xcc = inJs["xCpu"].asDouble();
    inl[idx] = icond(idx, xcc);
}

__host__ void mpi_type(MPI_Datatype *dtype)
{ 
    MPI_Datatype typs[2] = {MPI_R, MPI_R};
    int n[2] = {1};
    MPI_Aint disp[2] = {0, sizeof(REAL)};

    MPI_Type_create_struct(2, n, disp, typs, dtype);
    MPI_Type_commit(dtype);
}

__device__ __host__ 
void stepUpdate(states *state, int idx, int tstep)
{
    int otx = MODULA(tstep); //Modula is output place
    int itx = (otx^1); //Opposite in input place.
    state[idx].u[otx] = DIMS.n*(state[idx].u[itx]);
}
