/**
	Global variables that are specific to the equations being solved.
*/

#ifndef EULERGLOBALS_H
#define EULERGLOBALS_H



#ifndef REAL
    #define REAL            float
    #define REALtwo         float2
    #define REALthree       float3
    #define SQUAREROOT(x)   sqrtf(x)

    #define ZERO            0.0f
    #define QUARTER         0.25f
    #define HALF            0.5f
    #define ONE             1.f
    #define TWO             2.f
#else

    #define ZERO            0.0
    #define QUARTER         0.25
    #define HALF            0.5
    #define ONE             1.0
    #define TWO             2.0
    #define SQUAREROOT(x)   sqrt(x)
#endif

/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int base; // Length of node + stencils at end (4)
    int idxend; // Last index (number of spatial points - 1)
};

struct states{
    REALthree Q; // Full Step state variables
    REAL Pr; // First step Pressure ratio
	REALthree Qmid; // Midpoint state variables
	REAL Prmid;
};

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/

#ifdef __CUDA_ARCH__

    __constant__ REAL *dBound[2];
    __constant__ dimensions dimens;

#else

#endif