/**
    This file evaluates the Euler equations applied to the 1D Sod Shock Tube problem.  It demonstrates the numerical solution to this problem in parallel using the GPU. The solution procedure uses a second order finite volume scheme with a minmod limiter parameterized by the Pressure ratio at cells on a three point stencil.  The solution also uses a second-order in time (RK2 or midpoint) scheme.
    
    The problem may be evaluated in three ways: Classic, SharedGPU, and Hybrid.  Classic simply steps forward in time and calls the kernel once every timestep (predictor step or full step).  SharedGPU uses the GPU for all computation and applies the swept rule.  Hybrid applies the swept rule but computes the node on the boundary with the CPU.  
*/
/* 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

// Two primary strategies used in this code: global variables and templating using structures.

#include "decomposition/classicCore.h"
// #include "decomposition/sweptCore.h"

void exitMerge()
{
    system("json-merge path/to/jsons/*.json")
}

int main(int argc, char *argv[])
{   
    makeMPI(argc, &argv);

    if (!rank[1]) atexit(exitMerge);

    getDeviceInformation();

    states *state;
    double *xpts;

    std::ifstream injson(argv[1]);
    json inJ;
    injson >> inJ;
    injson.close();

    parseArgs(inJ, argc, &argv);
    initArgs(inJ);

    hBound[0] = {};
    hBound[1] = {};
    
    delegateDomain(double *xpts, states *state);

    for (int k=0; k<dv; k++) initialState(inJ, &state[k]->Q[0]);

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

    eCheckIn(dv, tpb, argc); //Initial error checking.

    // We always know that there's some eqConsts struct that we need to 
    // to put into constant memory.
    // PROBABLY NEED TO CHECK TO MAKE SURE THERE"S A GPU FIRST.
    if (gpuYes) cudaMemcpyToSymbol(deqConsts,&heqConsts,sizeof(eqConsts));

    // Start the counter and start the clock.
    MPI_Barrier(MPI_COMM_WORLD);
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
        tfm = classicWrapper(state, xpts);
    }

    endMPI();

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

    std::ofstream soljson(argv[2]);
    soljson << solution;

    if (rank == 0)
    {
        //READ OUT JSONS
        
        timed *= 1.e3;

        double n_timesteps = tfm/dt;

        double per_ts = timed/n_timesteps;

        cout << n_timesteps << " timesteps" << endl;
        cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

        json timing;
        timing[dv][tpb][gpuA] = per_ts;

        std::ofstream timejson(argv[2]);
        timejson << timing;
    }

    if (xgpu)
    {
        cudaDeviceSynchronize();

        cudaEventDestroy( start );
        cudaEventDestroy( stop );
        cudaDeviceReset();
    }

	return 0;
}