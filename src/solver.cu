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
//#include "decomposition/sweptCore.h"

#ifndef HDW
    #define HDW     "hardware/WORKSTATION.json"
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
    makeMPI(argc, argv);

    std::string ext = ".json";
    std::string myrank = std::to_string(ranks[1]);
    std::string sout = argv[3];
    sout.append(myrank);
    sout.append(ext); 
    std::string scheme = argv[1];

    std::ifstream hwjson(HDW, std::ifstream::in);
    json hwJ;
    hwjson >> hwJ;
    hwjson.close();

    std::vector<int> gpuvec = hwJ["GPU"];
    std::vector<int> smGpu(gpuvec.size());
    cGlob.nThreads = hwJ["nThreads"]; // Potetntial for non constant
    cGlob.hasGpu = gpuvec[ranks[1]];
    std::partial_sum(gpuvec.begin(), gpuvec.end(), smGpu.begin());
    cGlob.nGpu = smGpu.back();
    smGpu.insert(smGpu.begin(), 0);
    int gpuID = hwJ["gpuID"];
    
    // Equation, grid, affinity data
    std::ifstream injson(argv[1], std::ifstream::in);
    json inJ;
    injson >> inJ;
    injson.close();

    parseArgs(inJ, argc, argv);
    initArgs(inJ);

    /*  
        Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */

    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * cGlob.hasGpu * smGpu[ranks[1]]; //
    states **state;
    double **xpts;

    int exSpace = (!scheme.compare("S") ? cGlob.htp : 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int xalloc = xc + exSpace;

    if (cGlob.hasGpu)
    {
        cudaSetDevice(gpuID);
        
        state = new states* [3];
        xpts = new double* [3];
        cudaHostAlloc((void **) &xpts[0], xc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &xpts[1], (cGlob.xg + exSpace) * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &xpts[2], xc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[1], (cGlob.xg + exSpace) * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[2], xalloc * cGlob.szState, cudaHostAllocDefault);

        int pone = (strt + xc);
        int ptwo = (pone + cGlob.xg);

        for (int k=1; k <= xc; k++) 
        {
            initialState(inJ, k, strt, state[0], xpts[0]); 
            initialState(inJ, k, ptwo, state[2], xpts[2]); 
        }

        for (int k=1; k <= cGlob.xg; k++)  initialState(inJ, k, pone, state[1], xpts[1]); 

        cudaMemcpyToSymbol(&deqConsts, &heqConsts, sizeof(eqConsts));

        if (sizeof(REAL)>6) 
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
    }
    else 
    {
        state = new states* [1];
        xpts = new double* [1];
        cudaHostAlloc((void **) &xpts[0], xalloc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        for (int k=1; k<=xc; k++)  initialState(inJ, k, strt, state[0], xpts[0]); 
    }

    int tstep = 1;
    // Start the counter and start the clock.
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    // Call the correct function with the correct algorithm.
    double tfm;

    if (!scheme.compare("C"))
    {
        tfm = classicWrapper(state, xpts, &tstep);
    }
    else if  (!scheme.compare("S"))
    {
        tfm = sweptWrapper(state, xpts, &tstep);
    }
    else
    {
        std::cerr << "Incorrect or no scheme given" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timed, start, stop);

    endMPI();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    std::ofstream soljson(argv[3]);
    soljson << solution;
    soljson.close();

    if (!ranks[1])
    {
        //READ OUT JSONS
        
        timed *= 1.e3;

        double n_timesteps = tfm/cGlob.dt;

        double per_ts = timed/n_timesteps;

        std::cout << n_timesteps << " timesteps" << std::endl;
        std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

        json timing;
        timing[cGlob.nX][cGlob.tpb][cGlob.gpuA] = per_ts;

        std::ofstream timejson(argv[4]);
        timejson << timing;
        timejson.close();
    }

    if (cGlob.hasGpu)
    {
        cudaDeviceSynchronize();
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        for (int k=0; k<3; k++)
        {
            cudaFreeHost(xpts[k]);
            cudaFreeHost(state[k]);
        }
        
        delete[] xpts;
        delete[] state;
        cudaDeviceReset();
    }
    else
    {
        cudaFreeHost(xpts[0]);
        cudaFreeHost(state[0]);
        delete[] xpts;
        delete[] state;
    }
	return 0;
}