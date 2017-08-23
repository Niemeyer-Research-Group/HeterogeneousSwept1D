/**
	The equations specific global variables and function prototypes.
*/

#ifndef EULERCF_H
#define EULERCF_H

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

#include "myVectorTypes.h"
#include "json.hpp"

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

using json = nlohmann::json;

/*
//---------------// 
Means this functions is called from the primary program and therefore must be included BY NAME 
in any equation discretization handled by the software.
*/

__host__ REAL density(REALthree subj);

__host__ REAL velocity(REALthree subj);

__host__ REAL energy(REALthree subj);

__device__ __host__
__forceinline__ REAL pressure(REALthree qH);

__host__ REAL printout(const int i, states *state); //---------------//

__host__ void equationSpecificArgs(json inJ); //---------------//

__host__ void initialState(json inJ, int idx, int xst, states *inl, double *xs); //---------------//

__host__ void mpi_type(MPI_Datatype *dtype); //---------------//

__device__ __host__ 
__forceinline__ REAL pressureRoe(REALthree qH);

__device__ __host__ 
__forceinline__ void pressureRatio(states *state, int idx, int tstep);

__device__ __host__ 
__forceinline__ REALthree limitor(REALthree qH, REALthree qN, REAL pRatio);

__device__ __host__ 
__forceinline__ REALthree eulerFlux(REALthree qL, REALthree qR);

__device__ __host__ 
__forceinline__ REALthree eulerSpectral(REALthree qL, REALthree qR);

__device__ __host__ 
__forceinline__ void eulerStep(states *state, int idx, int tstep);

__device__ __host__  
void stepUpdate(states *state, int idx, int tstep); //---------------//



#endif
