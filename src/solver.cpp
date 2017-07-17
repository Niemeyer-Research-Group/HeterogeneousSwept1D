/**
    NOTE: This file is where the explanatory comments for this package appear. The other source files only have superficial comments.

    This file evaluates the Euler equations applied to the 1D Sod Shock Tube problem.  It demonstrates the numerical solution to this problem in parallel using the GPU. The solution procedure uses a second order finite volume scheme with a minmod limiter parameterized by the Pressure ratio at cells on a three point stencil.  The solution also uses a second-order in time (RK2 or midpoint) scheme.

    The boundary conditions are:
    Q(t=0,x) = QL if x<L/2 else QR
    Q(t,x=0,dx) = QL
    Q(t,x=L,L-dx) = QR
    Where Q is the vector of dependent variables.
    
    The problem may be evaluated in three ways: Classic, SharedGPU, and Hybrid.  Classic simply steps forward in time and calls the kernel once every timestep (predictor step or full step).  SharedGPU uses the GPU for all computation and applies the swept rule.  Hybrid applies the swept rule but computes the node on the boundary with the CPU.  
*/
/* 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "myVectorTypes.h"

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <yaml.h>


int main( int argc, char *argv[] )
{

    MPI_Datatype struct_type;
    mpi_type(&struct_type);
    
    // Set shared memory banks to double if REAL is double.
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    makeMPI(int argc, char* argv[]);

    dimz.gam = 1.4;
    dimz.mgam = 0.4;

    bd[0].x = ONE; //Density
    bd[1].x = 0.125;
    bd[0].y = ZERO; //Velocity
    bd[1].y = ZERO;
    bd[0].z = ONE/dimz.mgam; //Energy
    bd[1].z = 0.1/dimz.mgam;

    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const double dt = atof(argv[3]);
	const double tf = atof(argv[4]) - QUARTER*dt; //Finish time
    const double freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //2 for Alternate, 1 for GPUShared, 0 for Classic
    const int bks = dv/tpb; //The number of blocks
    const double dx = lx/((REAL)dv-TWO);
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4;
    dimz.idxend = dv-1;
    dimz.idxend_1 = dv-2;

    for (int k=-2; k<3; k++) dimz.hts[k+2] = (tpb/2) + k;
        
    cout << "Euler --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dimz.dt_dx << endl;

    eCheckIn(dv, tpb, argc); //Initial error checking.

    if (dimz.dt_dx > .21)
    {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.21 for stability." << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALthree *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALthree), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALthree), cudaHostAllocDefault);

	for (int k = 0; k<dv; k++) IC[k] = (k<dv/2) ? bd[0] : bd[1];

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[7],ios::trunc);
    fwr.precision(10);

	// Write out x length and then delta x and then delta t.
	// First item of each line is variable second is timestamp.
	fwr << lx << " " << (dv-2) << " " << dx << " " << endl;

    fwr << "Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << "Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << "Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALthree));

    // Start the counter and start the clock.
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    // Call the correct function with the correct algorithm.
    cout << scheme << " " ;
    double tfm;
    if (scheme)
    {
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, scheme-1, IC, T_final, freq, fwr);
    }
    else
    {
        tfm = classicWrapper(bks, tpb, dv, dt, tf, IC, T_final, freq, fwr);
    }

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    timed *= 1.e3;

    double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

    if (argc>7)
    {
        ofstream ftime;
        ftime.open(argv[8],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << "Density " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << "Energy " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << energy(T_final[k]) << " ";
    fwr << endl;

    fwr << "Pressure " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_final[k]) << " ";
    fwr << endl;

	fwr.close();

    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;
}