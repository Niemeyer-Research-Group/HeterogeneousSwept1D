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
#include <string>
#include "cudaUtils.h"
#include <vector>

// #include "matplotlib-cpp/matplotlibcpp.h"

// namespace plt = matplotlibcpp;

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


__host__ __device__
__forceinline__
int mod(int a, int b) { return (a % b + b) % b; }

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
    int nPts;
    __host__ __device__
    __forceinline__
    int modg(int gd) { return mod(gd, nPts);}
};

__constant__ discConstants disc;

template <int n>
struct ar{
    int a[n];
};

template <int n>
__device__
ar<n> idxes()
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int st = n/2;
    ar <n> ii;
    #pragma unroll
    for (int k=0; k<n; k++) ii.a[k] = disc.modg((gid-st) + k);
	return ii;
}

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
    void printme()
    {
        std::cout << "KS ---" << std::endl;
        std::cout << "#Blocks: " << bks << " | Length: " << lx << " | dt/dx: " << dt/dx << std::endl;
        std::cout << "TPB: " << tpb << " Grid Size " << dv << std::endl;
        std::cout << "--------------------" << std::endl;
    }
} rv;

#include "KSArray.hpp"
#ifdef PAD
    #include "KSAtomicPad.hpp"
#else
    #include "KSAtomic.hpp"
#endif

std::vector<double> tests(double *d1, states *s2, int dv, double tl)
{
	int t = 0;
	double absit;
    double summit   = 0;
    double sumd     = 0;
    std::vector <double> sto;

	for(int k=0; k<dv; k++)
	{
		absit   = (std::abs(d1[k] - s2[k].u[0]));
		t      += (absit > tl);
        summit += absit;
        sumd   += std::abs(d1[k]);
        sto.push_back(absit);
	}

	if (!t)
	{
		std::cout << "EQUAL! " << summit << " OUT OF " << sumd << std::endl;
	}
	else
	{
		std::cout << "NOT EQUAL! " << summit << " OUT OF " << sumd << std::endl;
		std::cout << t << " out of " << dv << std::endl;
    }
    return sto;
}

std::vector<double> tests(double *d1, double *s2, int dv, double tl)
{
	int t = 0;
	double absit;
    double summit = 0;
    double sumd = 0;
    std::vector <double> sto;

	for(int k=0; k<dv; k++)
	{
		absit   = (std::abs(d1[k] - s2[k]));
		t      += (absit > tl);
        summit += absit;
        sumd   += std::abs(d1[k]);
        sto.push_back(absit);
	}

	if (!t)
	{
		std::cout << "EQUAL! " << summit << " OUT-OF " << sumd << std::endl;
	}
	else
	{
		std::cout << "NOT EQUAL! " << summit << " OUT-OF " << sumd << std::endl;
		std::cout << t << " out of " << dv << std::endl;
    }
    return sto;
}

inline void writeTime(const cudaTime &Timer)
{
    if (Timer.nTimes > 10) return;

    std::ofstream timeOut;
    timeOut.open("atomicArray.csv", std::ios::app);
    timeOut.seekp(0, std::ios::end);
    bool starts = (timeOut.tellp() == 0);
    if (starts) timeOut << "tpb,dv,sweptArray,sweptAtomic,classicArray,classicAtomic\n";

    timeOut << rv.tpb << "," << rv.dv;
    for (int i=0; i<Timer.nTimes; i++) timeOut << "," << Timer.times[i];
    timeOut << std::endl;
    timeOut.close();
}


int main( int argc, char *argv[])
{

	cudaSetDevice(GPUNUM);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    rv.dv = atoi(argv[1]); //Number of spatial points
	rv.tpb = atoi(argv[2]); //Threads per Block
    rv.dt = atof(argv[3]); //delta T timestep
	rv.tf = atof(argv[4]) - 0.25*rv.dt; //Finish time
    rv.bks = rv.dv/rv.tpb; //The number of blocks
    rv.lx = rv.dv*dx;

    rv.printme();

    discConstants dsc = {
		ONE/(FOUR*dx), //dx
		ONE/(dx*dx), //dx^2
		ONE/(dx*dx*dx*dx), //dx^4
		rv.dt, //dt
		rv.dt*0.5, //dt half
		rv.tpb + 4, //length of row of shared array
		(rv.tpb+4)/2, //midpoint of shared array row
		rv.dv-1, //last global thread id
        rv.dv
    };

    std::vector<double> vatom;
    std::vector<double> varray;
    std::vector<double> xa;

    cudaMemcpyToSymbol(disc,&dsc,sizeof(dsc));

    cudaTime *timer = new cudaTime();
    double *arrayState, *permState, *tradeState;
    states *atomicState;
    double tfm;
    double tol=1.0e-3;

    const int arSize = sizeof(double) * rv.dv;
    const int atomSize = sizeof(states) * rv.dv;

    cudaHostAlloc((void **) &permState, arSize, cudaHostAllocDefault);
    cudaHostAlloc((void **) &arrayState, arSize, cudaHostAllocDefault);
    cudaHostAlloc((void **) &atomicState, atomSize, cudaHostAllocDefault);
    cudaHostAlloc((void **) &tradeState, arSize, cudaHostAllocDefault);

    double idou, nTimes;
    for (int k=0; k<rv.dv; k++)
    {
        xa.push_back(k*dx);
        idou                = initFun((double)k*dx);
        permState[k]        = idou;
        arrayState[k]       = idou;
        atomicState[k].u[0] = idou;
    }

    tests(arrayState, atomicState, rv.dv, tol);

    //CLASSIC TEST
    std::cout << " SWEPT TEST " << std::endl;

    timer->tinit();
    tfm = sweptWrapper(arrayState);
    nTimes = tfm/rv.dt;
    std::cout << tfm << "  " << nTimes << std::endl;
    timer->tfinal(nTimes);

    cudaKernelCheck();

    timer->tinit();
    tfm = sweptWrapper(atomicState);
    nTimes = tfm/rv.dt;
    std::cout << tfm << "  " << nTimes << std::endl;
    timer->tfinal(nTimes);

    cudaKernelCheck();

    std::vector <double> ipon = tests(arrayState, atomicState, rv.dv, tol);

    rv.tf = tfm-(0.25*rv.dt); // RESET IT

    // plt::named_plot("Differences", xa, ipon, "-r");
    // plt::legend();
    // plt::show();

    std::cout << *timer << std::endl;

    std::cout << " ----------------------- " << std::endl;
    std::cout << " CLASSIC TEST " << std::endl;

    for (int k=0; k<rv.dv; k++) 
    {
        varray.push_back(arrayState[k]);
        vatom.push_back(atomicState[k].u[0]);
        tradeState[k]        = arrayState[k];
        arrayState[k]       = permState[k];
    }

    // plt::named_plot("Atomic", xa, vatom, "-r");
    // plt::named_plot("Array", xa, varray, "-b");
    // plt::title("Classic");
    // plt::legend();
    // plt::show();
    
    timer->tinit();
    tfm = classicWrapper(arrayState);
    nTimes = tfm/rv.dt;
    std::cout << tfm << "  " << nTimes << std::endl;
    timer->tfinal(nTimes);

    cudaKernelCheck();

    std::cout << " Swept vs Classic ARRAY Agreement: "; 
    tests(tradeState, arrayState, rv.dv, tol);

    for (int k=0; k<rv.dv; k++) 
    {
        tradeState[k]        = atomicState[k].u[0];
        atomicState[k].u[0]  = permState[k];
    }
    
    timer->tinit();
    tfm = classicWrapper(atomicState);
    nTimes = tfm/rv.dt;
    std::cout << tfm << "  " << nTimes << std::endl;
    timer->tfinal(nTimes);

    cudaKernelCheck();
    std::cout << " Swept vs Classic ATOMIC Agreement: "; 
    tests(tradeState, atomicState, rv.dv, tol);

    ipon = tests(arrayState, atomicState, rv.dv, tol);
    // plt::named_plot("Differences", xa, ipon, "-r");
    // plt::legend();
    // plt::show();

    std::cout << *timer << std::endl;
    for (int k = 0; k<rv.dv; k++)
    {
        varray[k] = arrayState[k];
        vatom[k]  = atomicState[k].u[0];
    }

    if (argc>5) writeTime(*timer);

    // plt::named_plot("Atomic", xa, vatom, "-r");
    // plt::named_plot("Array", xa, varray, "-b");
    // plt::title("Swept");
    // plt::legend();
    // plt::show();

    std::cout << " ----------------------- " << std::endl;

    cudaFreeHost(arrayState);
    cudaFreeHost(atomicState);
    cudaFreeHost(permState);

    delete timer;

    return 0;
}