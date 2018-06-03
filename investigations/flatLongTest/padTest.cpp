/**
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp --ptxas-options=-v

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <ostream>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>

#ifndef GPUNUM
    #define GPUNUM              0
#endif

#define QUARTER     0.25
#define HALF        0.5
#define ONE         1.0
#define TWO         2.0
#define FOUR        4.0
#define SIX			6.0

#define NSTEPS              4 // How many substeps in timestep.

// Since anyone would need to write a header and functions file, why not just hardwire this.
// If the user's number of steps isn't a power of 2 use the other one.

#define MODULA(x)           x & (NSTEPS-1)
// #define MODULA(x)           x % NSTEPS

#define DIVMOD(x)           (MODULA(x)) >> 1

const double dx = 0.5;

struct discConstants{

	double dxTimes4;
	double dx2;
	double dx4;
	double dt;
	double dt_half;
	int base;
	int ht;
    int idxend;
};

__constant__ discConstants disc;

//Initial condition.
__host__
double initFun(double xnode)
{
	return TWO * cos(19.0*xnode*M_PI/128.0);
}

__device__ __forceinline__
double fourthDer(double tfarLeft, double tLeft, double tCenter, double tRight, double tfarRight)
{
	return (tfarLeft - FOUR*tLeft + SIX*tCenter - FOUR*tRight + tfarRight)*(disc.dx4);
}

__device__ __forceinline__
double secondDer(double tLeft, double tRight, double tCenter)
{
	return (tLeft + tRight - TWO*tCenter)*(disc.dx2);
}

__device__ __forceinline__
double convect(double tLeft, double tRight)
{
	return (tRight*tRight - tLeft*tLeft)*(disc.dxTimes4);
}

struct runVar{
    int bks, tpb, dv;
    double dt, tf, lx;
    size_t smem;

    void printme()
    {
        cout << "KS ---" << endl;
        cout << "#Blocks: " << bks << " | Length: " << lx << " | dt/dx: " << dt/dx << endl;
        cout << "TPB: " << tpb << " Grid Size " << dv << endl;
    }
} rv;

#include "KSAtomic.hpp"

int main( int argc, char *argv[])
{

	cudaSetDevice(GPUNUM);
	// if cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    rv.dv = atoi(argv[1]); //Number of spatial points
	rv.tpb = atoi(argv[2]); //Threads per Block
    rv.dt = atof(argv[3]); //delta T timestep
	rv.tf = atof(argv[4]) - 0.25*rv.dt; //Finish time
    rv.bks = rv.dv/rv.tpb; //The number of blocks
    rv.lx = rv.dv*dx;
    rv.smem = (2*rv.tpb+8)*szSt;

    rv.printme();

    discConstants dsc = {
		ONE/(FOUR*dx), //dx
		ONE/(dx*dx), //dx^2
		ONE/(dx*dx*dx*dx), //dx^4
		rv.dt, //dt
		rv.dt*0.5, //dt half
		rv.tpb + 4, //length of row of shared array
		(rv.tpb+4)/2, //midpoint of shared array row
		rv.dv-1 //last global thread id
	};

    cudaMemcpyToSymbol(disc,&dsc,sizeof(dsc));

    double *arrayState;
    states *atomicState;

    // Initialize, time, write out.  Compare Array to Atomic.
    // YOU HAVE 1 HOUR.

    cudaFreeHost(arrayState);
    cudaFreeHost(atomicState);

    return 0;

}