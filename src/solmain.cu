
#include <fstream>
#include "euler.h"
#include "decomp.h"
#include "classic.h"
#include "swept.h"
#include <iomanip>

/**
----------------------
    MAIN PART
----------------------
*/

#ifndef HDW
    #define HDW     "WORKSTATION.json"
#endif

using std::cout, std::endl, std::vector, std::string, std::ifstream, std::ostream;

vector<int> jsonP(jsons jp, size_t sz)
{
	vector <int> outv;
	for(int i=0; i<sz; i++)
	{
		outv.push_back(jp[i].asInt());
	}
	return outv;
}

int main(int argc, char *argv[])
{   
    makeMPI(argc, argv);

    string ext = ".json";
    string myrank = std::to_string(ranks[1]);
    string sout = argv[3];
    sout.append(myrank);
    sout.append(ext); 
    string scheme = argv[1];

    ifstream hwjson(HDW, ifstream::in);
    jsons hwJ;
    hwjson >> hwJ;
    hwjson.close();

    vector<int> gpuvec = jsonP(hwJ["GPU"], 1);
    vector<int> smGpu(gpuvec.size());
    vector<int> threadv =  jsonP(hwJ["nThreads"], 1);
    cGlob.nThreads=threadv[ranks[1]]; // Potetntial for non constant
    cGlob.hasGpu = gpuvec[ranks[1]];
    std::partial_sum(gpuvec.begin(), gpuvec.end(), smGpu.begin());
    cGlob.nGpu = smGpu.back();
    smGpu.insert(smGpu.begin(), 0);
    vector <int> myGPU = jsonP(hwJ["gpuID"], 1);
    int gpuID = myGPU[ranks[1]];
    
    // Equation, grid, affinity data
    ifstream injson(argv[2], ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();

    /*  
        Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */

    int exSpace = (scheme.compare("S")) ? cGlob.htp : 2;
    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * cGlob.hasGpu * smGpu[ranks[1]]; 
    states *state;
    vector <int> xpts;
    xpts.push_back(strt-1) // Basic vector will have all pass, start, and split points.
    //The pattern is important. 

    int mon;

    if (cGlob.hasGpu)
    {
        //GPU set up. Which device, what precision, copy constants to device.
        cudaSetDevice(gpuID);
        if (sizeof(REAL)>6) 
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts))

        // Add the other half of the CPU and the GPU alloc.
        xalloc = exSpace + cGlob.xg + cGlob.xcpu;
        cudaMallocManaged((void **) &state, xalloc * cGlob.szState);

        for (int k=0; k <= xalloc; k++)  initialState(inJ, k + strt, state);
        for (int k=1; k<=xalloc; k++) solutionOutput(state, 0.0, xpts);                  
    }
    else 
    {
        xalloc = exSpace + cGlob.xcpu;
        malloc((void **) &state, xalloc * cGlob.szState);
        for (int k=0; k <= xalloc; k++)  initialState(inJ, k + strt, state);  
        for (int k=1; k<=xalloc-cGlob.ht; k++) solutionOutput(state, 0.0, xpts);
    }

    int tstep = 1;
    // Start the counter and start the clock.  Maybe should time it with MPI.  Still use cudaSynchronize for GPU nodes.
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
        cerr << "Incorrect or no scheme given" << endl;
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

    int nomar;
    ofstream soljson(argv[3]);
    soljson << solution;
    soljson.close();

    if (!ranks[1])
    {
        //READ OUT JSONS
        
        timed *= 1.e3;

        double n_timesteps = tfm/cGlob.dt;

        double per_ts = timed/n_timesteps;

        cout << n_timesteps << " timesteps" << endl;
        cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

        // Equation, grid, affinity data
        try {
            ifstream tjson(argv[4], ifstream::in);
            tjson >> timing;
            tjson.close();
        }
        catch (...) {}

        string tpbs = std::to_string(cGlob.tpb);
        string nXs = std::to_string(cGlob.nX);
        string gpuAs = std::to_string(cGlob.gpuA);
        cout << cGlob.gpuA << endl;

        ofstream timejson(argv[4], ofstream::trunc);
        timing[tpbs][nXs][gpuAs] = per_ts;
        timejson << timing;
        timejson.close();
    }

    if (cGlob.hasGpu)
    {
        cudaDeviceSynchronize();
        cudaEventDestroy( start );
        cudaEventDestroy( stop );
        cudaFree(state);
        cudaDeviceReset();
    }
    else
    {
        free(state);
    }
	return 0;
}
