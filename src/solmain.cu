
#include <fstream>

#define cudaCheckError(ans) { cudaCheck((ans), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t code, const char *file, int line, bool abort=false) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include "eulerCF/eulerCF.h"
#include "decomp.h"
#include "classic.h"
// #include "swept.h"

/**
----------------------
    MAIN PART
----------------------
*/

#ifndef HDW
    #define HDW     "hardware/WORKSTATION.json"
#endif

using namespace std;

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

    int exSpace = ((int)!scheme.compare("S") * cGlob.ht) + 2;
    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * smGpu[ranks[1]]; 
    int xalloc, xwrt;
    states *state;

    int mon;
    cout << "Made it to allocation flow " << endl;

    if (cGlob.hasGpu)
    {
        //GPU set up. Which device, what precision, copy constants to device.
        cudaSetDevice(gpuID);
        if (sizeof(REAL)>6) 
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }

        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

        // Add the other half of the CPU and the GPU alloc.
        xwrt = cGlob.xg + cGlob.xcpu;
        xalloc = exSpace + xwrt;
        cout << "Num pts: " << xalloc << endl;
        cudaCheckError(cudaMallocManaged((void **) &state, xalloc * cGlob.szState));              
    }
    else 
    {
        xwrt = cGlob.xcpu;
        xalloc = exSpace + xwrt;
        cudaCheckError(cudaHostAlloc((void **) &state, xalloc * cGlob.szState, cudaHostAllocDefault));
    }
    
    for (int k=0; k <= xalloc; k++) initialState(inJ, state, k, strt);
    for (int k=1; k <= xwrt; k++) solutionOutput(state, 0.0, k, strt); 

    int tstep = 1;
    double timed, tfm;

    MPI_Barrier(MPI_COMM_WORLD);
    if (!ranks[1]) timed = MPI_Wtime();
    cout << "Made it to Calling the function " << endl;

    // Call the correct function with the correct algorithm.
    if (!scheme.compare("C"))
    {
        tfm = classicWrapper(state, strt, &tstep);
    }
    else if  (!scheme.compare("S"))
    {
        //tfm = sweptWrapper(state, strt, &tstep);
    }
    else
    {
        cerr << "Incorrect or no scheme given" << endl;
    }

    if (cGlob.hasGpu)
    {
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (!ranks[1]) timed = (MPI_Wtime() - timed);

    // Print out final solution.
    #pragma omp parallel for
    for (int k=1; k <= xwrt; k++) solutionOutput(state, tfm, k, strt); 
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
        timed *= 1.e6;

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
        cudaFree(state);
        cudaDeviceReset();
    }
    else
    {
        cudaFreeHost(state);
    }
	return 0;
}
