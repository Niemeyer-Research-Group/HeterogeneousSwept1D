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
#define INDEXER(x, y, nx)	y*nx + x


//---------------//
struct states{
    REAL u[2];
    //size_t tstep; // Consider as padding.
};

std::string fspec = "Wave";
std::string outVars[NVARS] = {"Velocity"}; //---------------//

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

struct gridsteps{
	double x, y, t;
};

void mpi_type(MPI_Datatype *dtype)
{
    MPI_Type_contiguous(2, MPI_R, dtype);
    MPI_Type_commit(dtype);
}

// This struct is going in constant memory.
struct constWave
{
	gridsteps gs;
	uint nx, ny;
	double cx, cy, homecoeff;

	void init(gridsteps g, uint nx, uint ny, double c)
	{
		gs = g;
		ny = ny;
		nx = nx;
		cx = c*gs.t/gs.x;
		cx *= cx;
		cy = c*gs.t/gs.y;
		cy *= cy;
		homecoeff = 2.0-2.0*(cx+cy);
	}
};

constWave HCONST;
__constant__ constWave DCONST;

void initState(states *state, const int x, const int y)
{
	static int nx = HCONST.nx;
	static int ny = HCONST.ny;
	for (int k=0; k<2; k++) state->u[k] = std::exp(-50.0 * 
			std::sqrt((x-nx*0.5) * (x-nx*0.5) + 
			(y-ny*0.5) * (y-ny*0.5)));
}

REAL printout(states *state)
{
	return state->u[0];
}

#ifdef __CUDA_ARCH__
    #define A			DCONST
#else
    #define A    		HCONST
#endif

__host__ __device__
void stepUpdate(states *state, const int idx,  const int idy, const int tstep)
{
	const int ix = INDEXER(idx, idy, A.nx);
	const int itx = tstep % NSTEPS; //Modula is output place
	const int otx = (itx^1); //Opposite in input place.

	state[ix].u[otx] =  A.homecoeff * state[ix].u[itx] - state[ix].u[otx] + A.cx * (state[ix+1].u[itx] + state[ix-1].u[itx]) + A.cy * (state[ix+A.nx].u[itx] + state[ix-A.nx].u[itx]);
}