// ALLONE

// Well we definitely need to get rid of the xpts.  Really I need to concentrate on getting the output right so I can check the answers.  Then, if they're right, we can worry about streamlining this. Partly main problem, the keys in the output json are strings. Could read each in and then make it a data frame from dict.

#include <fstream>

#define cudaCheckError(ans) { cudaCheck((ans), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t code, const char *file, int line, bool abort=false) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include "heads.h"
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
    // Well we definitely need to get rid of the xpts.  Really I need to concentrate on getting the output right so I can check the answers.  Then, if they're right, we can worry about streamlining this. 
    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * cGlob.hasGpu * smGpu[ranks[1]]; 
    states **state;
    double **xpts;

    int exSpace = ((int)!scheme.compare("S") * cGlob.ht) + 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int xalloc = xc + exSpace;
    std::string pth = string(argv[3]);

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

        std::cout << "Str compare " << scheme.compare("S") << std::endl;
        std::cout << "realcpu: " << cGlob.xcpu << " SCHEME " << scheme<< std::endl;
        std::cout << "xc: " << xc << " extra space: " << exSpace << std::endl;
        std::cout << "gpupts: " << cGlob.xg + exSpace << " cpuPts: " << xalloc << std::endl;
        int pone = (strt + xc);
        int ptwo = (pone + cGlob.xg);
        std::cout << pone << " " << ptwo << std::endl;

        for (int k=0; k <= xalloc; k++) 
        {
            initialState(inJ, k, strt, state[0], xpts[0]); 
            initialState(inJ, k, ptwo, state[2], xpts[2]); 
        }

        for (int k=0; k <= cGlob.xg+1; k++)  initialState(inJ, k, pone, state[1], xpts[1]); 
        std::cout << ranks[1] << " " << strt << " " << pone << " " << ptwo << " " << cGlob.hasGpu << std::endl;

        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

        if (sizeof(REAL)>6) 
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
        for (int k=1; k<=xc; k++) solutionOutput(state[0]+k, 0.0, xpts[0][k]);               
        for (int k=1; k<=xc; k++) solutionOutput(state[2]+k, 0.0, xpts[2][k]);  
        for (int k=1; k<=cGlob.xg; k++) solutionOutput(state[1]+k, 0.0, xpts[1][k]);   
    }
    else 
    {
        state = new states* [1];
        xpts = new double* [1];
        cudaHostAlloc((void **) &xpts[0], xalloc * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        for (int k=0; k<=xalloc; k++) initialState(inJ, k, strt, state[0], xpts[0]); 
        for (int k=1; k<=xc; k++) solutionOutput(state[0]+k, 0.0, xpts[0][k]);     
    }

    // Check CUDA alloc.
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // If you have selected scheme I, it will only initialize and output the initial values.
    if (scheme.compare("I"))
    {
        int tstep = 1;
        double timed, tfm;

        MPI_Barrier(MPI_COMM_WORLD);
        if (!ranks[1]) timed = MPI_Wtime();
        cout << "Made it to Calling the function " << endl;

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

        if (cGlob.hasGpu)
        {
            cudaDeviceSynchronize();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (!ranks[1]) timed = (MPI_Wtime() - timed);

        for (int k=1; k<=xc; k++) solutionOutput(state[0]+k, tfm, xpts[0][k]);               
        for (int k=1; k<=xc; k++) solutionOutput(state[2]+k, tfm, xpts[2][k]);  
        for (int k=1; k<=cGlob.xg; k++) solutionOutput(state[1]+k, tfm, xpts[1][k]);   

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        if (!ranks[1])
        {
            //READ OUT JSONS
            
            timed *= 1.e6;

            double n_timesteps = tfm/cGlob.dt;

            double per_ts = timed/n_timesteps;

            std::cout << n_timesteps << " timesteps" << std::endl;
            std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

            // Equation, grid, affinity data
            try {
                std::ifstream tjson(argv[4], std::ifstream::in);
                tjson >> timing;
                tjson.close();
            }
            catch (...) {}

            std::string tpbs = std::to_string(cGlob.tpb);
            std::string nXs = std::to_string(cGlob.nX);
            std::string gpuAs = std::to_string(cGlob.gpuA);
            std::cout << cGlob.gpuA << std::endl;

            std::string spath = pth + "/t" + fspec + ext;
            std::ofstream timejson(spath.c_str(), std::ofstream::trunc);
            timing[tpbs][nXs][gpuAs] = per_ts;
            timejson << timing;
            timejson.close();
        }
    }

    std::string spath = pth + "/s" + fspec + "_" + std::to_string(ranks[1]) + ext;
    std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
    if (!ranks[1]) solution["meta"] = inJ;
    soljson << solution;
    soljson.close();

    endMPI();

    if (cGlob.hasGpu)
    {
        cudaDeviceSynchronize();

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
