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

// Use: mpirun --bind-to-socket exe [args]

#include "decomposition/classicCore.h"
#include "decomposition/sweptCore.h"

#ifndef HDW
    #define HDW     #@WORKSTATION.json
#endif

/*
    TOOD
    - Swept always passes so what to do about bCond.
    - Make sure all struct variables are correctly initialized
    - Watch cluster video, try to run something like the Test bench
    - Write json for workstation situation.
    - Write the cluster explorer code.
    - Using an npm js solution for merging is a bad idea, try something else.
*/

// // This feels like a bad idea.
// void exitMerge()
// {
//     system("json-merge path/to/jsons/*.json")
// }

int main(int argc, char *argv[])
{   
    makeMPI(argc, &argv);

    // if (!ranks[1]) atexit(exitMerge);

    // Maybe using my new script and imporitng another json instead of this.

    // getDeviceInformation();

    // Maybe not declaring and mallocing right now.

    // Hardware information in json. Use hwloc, lspci, lstopo etc to get it.

    std::ifstream hwjson(HDW, std::ifstream::in);
    json hwJ;
    hwjson >> hwJ;
    hwjson.close();

    std::vector<int> gpuvec = hwJ["GPU"];
    std::vector<int smGpu(ivec.size())
    cGlob.nThreads = hwJ["nThreads"]; // Potetntial for non constant
    cGlob.hasGPU = ivec[ranks[1]];
    std::partial_sum(ivec.begin(), ivec.end(), smGpu.begin());
    cGlob.nGpu = smGpu.back();
    smGpu.insert(smGpu.begin(), 0);
    
    // Equation, grid, affinity data
    std::ifstream injson(argv[1], std::ifstream::in));
    json inJ;
    injson >> inJ;
    injson.close();

    parseArgs(inJ, argc, &argv);
    initArgs(inJ);

    /*  Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */


    states **state;
    double **xpts;
    if (cGlob.hasGpu)
    {
        **state = new state* [3];
        **xpts = new double* [3];
        // Now you have the index in smGpu[rank]*xg + xcp*rank  so get the k value with dx.

    }
    else
    {

    }

    int prt = 1 + 2*cGlob.hasGpu;
    

    


    int strt, npt, nar;


    for (int k=0; k<rank[2]; k++) 
    {

    }
         


    for (int k=0; k<; k++) 
        initialState(inJ, &state[k]->Q[0]);

    // We always know that there's some eqConsts struct that we need to 
    // to put into constant memory.
    // PROBABLY NEED TO CHECK TO MAKE SURE THERE"S A GPU FIRST.


    if (xgpu) cudaMemcpyToSymbol(deqConsts, heqConsts, sizeof(eqConsts));

    int tstep = 1;
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
        tfm = classicWrapper(state, xpts, &tstep);
    }

    endMPI();

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timed, start, stop);

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
        timejson << std::setw(4) << timing << std::endl;
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