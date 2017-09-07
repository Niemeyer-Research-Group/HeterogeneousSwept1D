

#include "euler.h"
#include "decomp.h"
#include "classic.h"
#include "swept.h"

/**
----------------------
    MAIN PART
----------------------
*/

#ifndef HDW
    #define HDW     "WORKSTATION.json"
#endif

std::vector<int> jsonP(jsons jp, size_t sz)
{
	std::vector <int> outv;
	for(int i=0; i<sz; i++)
	{
		outv.push_back(jp[i].asInt());
	}
	return outv;
}

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
    jsons hwJ;
    hwjson >> hwJ;
    hwjson.close();

    std::vector<int> gpuvec = jsonP(hwJ["GPU"], 1);
    std::vector<int> smGpu(gpuvec.size());
    std::vector<int> threadv =  jsonP(hwJ["nThreads"], 1);
    cGlob.nThreads=threadv[ranks[1]]; // Potetntial for non constant
    cGlob.hasGpu = gpuvec[ranks[1]];
    std::partial_sum(gpuvec.begin(), gpuvec.end(), smGpu.begin());
    cGlob.nGpu = smGpu.back();
    smGpu.insert(smGpu.begin(), 0);
    std::vector <int> myGPU = jsonP(hwJ["gpuID"], 1);
    int gpuID = myGPU[ranks[1]];
    
    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();

    /*  
        Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */

    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * cGlob.hasGpu * smGpu[ranks[1]]; //
    states **state;
    double **xpts;

    int exSpace = (!scheme.compare("S")) ? cGlob.htp : 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int xalloc = xc + exSpace;
    int mon;

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
        std::cout << state[1][5].Q[0].x << std::endl;
        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

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

        std::string tpbs = std::to_string(cGlob.tpb);
        std::string nXs = std::to_string(cGlob.nX);
        std::string gpuAs = std::to_string(cGlob.gpuA);
        std::cout << cGlob.gpuA << std::endl;
        timing[nXs][tpbs][gpuAs] = per_ts;

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
