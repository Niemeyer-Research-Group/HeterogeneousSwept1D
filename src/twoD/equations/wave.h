/**
	The equations specific global variables.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <mpi.h>

#include <iostream>
#include <ostream>
#include <istream>
#include <fstream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

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
#define INDEXER(x, y, nx)	y*nx + x


//---------------//
struct states{
    REAL u[2];
    //size_t tstep; // Consider as padding.
};
typedef Json::Value jsons;
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

void mpi_type(MPI_Datatype *dtype)
{
    MPI_Type_contiguous(2, MPI_R, dtype);
    MPI_Type_commit(dtype);
}

// This struct is going in constant memory.
// Could probably use Region dimensions in this.
struct constWave
{
	int nx, ny;
	int blockSide, blockBase;
	double cx, cy, homecoeff;

	void init(jsons inJ)
	{
		double dx = inJ["dx"].asDouble();
		double dy = inJ["dy"].asDouble();
		double dt = inJ["dt"].asDouble();
		double cfl = inJ["cfl"].asDouble(); //Some way to give c and get cfl
		ny = inJ["nY"].asDouble();
		nx = inJ["nX"].asDouble();
		blockSide = inJ['blockSide'].asDouble();
		blockBase = blockSide+2; 

		// CFL given should be max cfl for all dimensions.
		double c = cfl * std::min(dx,dy)/dt;

		cx = c*dt/dx;
		cx *= cx;
		cy = c*dt/dy;
		cy *= cy;
		homecoeff = 2.0-2.0*(cx+cy);
	}
};

constWave HCONST;
__constant__ constWave DCONST;

#ifdef __CUDA_ARCH__
    #define A			DCONST
#else
    #define A    		HCONST
#endif

__host__
void initState(states *state, const int x, const int y)	
{
	double fillVal = std::exp(-50.0 * std::sqrt((x-A.nx*0.5) * (x-A.nx*0.5) + (y-A.ny*0.5) * (y-A.ny*0.5)));

	for (int k=0; k<2; k++) state->u[k] = fillVal;
};

__host__ __device__
double printout(states *state, int i) {return state->u[0];}

__host__ __device__
void stepUpdate(states **state, const int idx,  const int idy, const int tstep)
{
	const int otx = MODULA(tstep); //Modula is output place
	const int itx = (otx^1); //Opposite in input place.

	state[idy][idx].u[otx] =  A.homecoeff * state[idy][idx].u[itx] - state[idy][idx].u[otx] + A.cx * (state[idy][idx+1].u[itx] + state[idy][idx-1].u[itx]) + A.cy * (state[idy+1][idx].u[itx] + state[idy-1][idx].u[itx]);
}