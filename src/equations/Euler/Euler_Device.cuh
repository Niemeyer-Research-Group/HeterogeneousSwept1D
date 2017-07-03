/**
	Functions for the discretization of the equation.
*/

#ifndef EULERDEVICE_CUH
#define EULERDEVICE_CUH

#include "EulerGlobals.h"

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

__host__ __device__ REAL pressure(REALthree current);

__host__ __device__ REAL pressureRatio();

#endif