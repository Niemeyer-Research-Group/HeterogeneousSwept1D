/**
	The equations specific global variables and function prototypes.
*/

#ifndef EULERCF_H
#define EULERCF_H

#include <cuda.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "myVectorTypes.h"

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

// Since anyone would need to write a header and functions file, why not just hardwire this.  If the user's number of steps isn't a power of 2 use the other one.

// #define MODULA(x)           x & (NSTEPS-1)  
// #define MODULA(x)           x % NSTEPS  

#define DIVMOD(x)           (x & (NSTEPS-1)) >> 1   

/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

// We don't need the grid data anymore because the node shapes are much simpler.  And do not affect the discretization implementation, only the decomposition.  Well the structs aren't accessible like arrays so shit.

struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
};

struct states{
    REALthree Q[2]; // Full Step, Midpoint step state variables
    REAL Pr; // Pressure ratio
};

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ dimensions dDimens; 
REALthree hBound[2];
dimensions hDimens;

 using json = nlohmann::json;
/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

__host__ void mpi_type(MPI_Datatype *dtype);

__host__ REAL density(REALthree subj);

__host__ REAL velocity(REALthree subj);

__host__ REAL energy(REALthree subj);

__host__ REALthree initialconditions();  // How to generalize this so the same inputs are valid for all equations.

__host__ __device__ REAL pressure(REALthree qH);

__host__ __device__ REAL pressureRoe(REALthree qH);

__host__ __device__ void pressureRatio(states *state, int idx, int tstep);

__host__ __device__ REALthree limitor(REALthree qH, REALthree qN, REAL pRatio);

__host__ __device__ REALthree eulerFlux(REALthree qL, REALthree qR);

__host__ __device__ REALthree eulerSpectral(REALthree qL, REALthree qR);

__host__ __device__ void eulerStep(states *state, int idx, int tstep);

__host__ __device__ void stepUpdate(states *state, int idx, int tstep);


// PERHAPS: Now use nvstd::functional to assign stepUpdate to the kernel AND MPI function!  

// #ifndef REAL
//     #define REAL            float
//     #define REALtwo         float2
//     #define REALthree       float3
//     #define SQUAREROOT(x)   sqrtf(x)

//     #define ZERO            0.0f
//     #define QUARTER         0.25f
//     #define HALF            0.5f
//     #define ONE             1.f
//     #define TWO             2.f
// #else

//     #define ZERO            0.0
//     #define QUARTER         0.25
//     #define HALF            0.5
//     #define ONE             1.0
//     #define TWO             2.0
//     #define SQUAREROOT(x)   sqrt(x)
// #endif

#endif
