/*
    Entry point for hsweep.
*/
#include <fstream>


#include "heads.h"
#include "decomp.h"
#include "classic.h"
#include "swept.h"
#include <unistd.h>

/**
----------------------
    MAIN PART
----------------------
*/

int main(int argc, char *argv[])
{
    makeMPI(argc, argv);

    #ifdef NOS
        if (!ranks[1]) std::cout << "No Solution Version." << std::endl;
    #endif

    std::string i_ext = ".json";
    std::string t_ext = ".csv";
    std::string myrank = std::to_string(ranks[1]);
    std::string scheme = argv[1];

    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();

    int prevGpu=0; //Get the number of GPUs in front of the current process.
    int gpuPlaces[nprocs]; //Array of 1 or 0 for number of GPUs assigned to process

    //If there are no GPUs or if the GPU Affinity is 0, this block is unnecessary.
    if (cGlob.nGpu > 0)
    {
        MPI_Allgather(&cGlob.hasGpu, 1, MPI_INT, &gpuPlaces[0], 1, MPI_INT, MPI_COMM_WORLD);
        for (int k=0; k<ranks[1]; k++) prevGpu+=gpuPlaces[k];
    }

    int strt = cGlob.xcpu * ranks[1] + cGlob.xg * prevGpu;
    cGlob.strt = strt;
    states **state;

    int exSpace = ((int)!scheme.compare("S") * cGlob.ht) + 2;
    int xc = (cGlob.hasGpu) ? cGlob.xcpu/2 : cGlob.xcpu;
    int nrows = (cGlob.hasGpu) ? 3 : 1;
    int xalloc = xc + exSpace;

    std::string pth = string(argv[3]);
    std::vector<int> xpts(1, strt); //Index at which pointer array starts.
    std::vector<int> alen(1, xc + 1); //Write out | Init args vector

    if (cGlob.hasGpu)
    {
        state = new states* [3];
        cudaHostAlloc((void **) &state[0], xalloc * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[1], (cGlob.xg + exSpace) * cGlob.szState, cudaHostAllocDefault);
        cudaHostAlloc((void **) &state[2], xalloc * cGlob.szState, cudaHostAllocDefault);

        cout << "Rank: " << ranks[1] << " has a GPU. gridpt Allocation: " << 2*xalloc + cGlob.xg * exSpace << endl;

        xpts.push_back(strt + xc);
        alen.push_back(cGlob.xg + 1);
        xpts.push_back(strt + xc + cGlob.xg);
        alen.push_back(xc + 1);

        cudaMemcpyToSymbol(deqConsts, &heqConsts, sizeof(eqConsts));

        if (sizeof(REAL)>6)
        {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
    }
    else
    {
        state = new states*[1];
        state[0] = new states[xalloc * cGlob.szState];
        cout << "Rank: " << ranks[1] << " no GPU. gridpt Allocation: " << xalloc << endl;
    }

    for (int i=0; i<nrows; i++)
    {
        for (int k=0; k<=alen[i]; k++)  initialState(inJ, state[i], k, xpts[i]-1);
        for (int k=1; k<alen[i]; k++)  solutionOutput(state[i], 0.0, k, xpts[i]);
    }

    // If you have selected scheme I, it will only initialize and output the initial values.

    if (scheme.compare("I"))
    {
        int tstep = 1;
        double timed, tfm;

		if (!ranks[1])
		{
            printf ("Scheme: %s - Grid Size: %d - Affinity: %.2f\n", scheme.c_str(), cGlob.nX, cGlob.gpuA);
            printf ("threads/blk: %d - timesteps: %.2f\n", cGlob.tpb, cGlob.tf/cGlob.dt);
		}

        MPI_Barrier(MPI_COMM_WORLD);
        if (!ranks[1]) timed = MPI_Wtime();

        if (!scheme.compare("C"))
        {
            tfm = classicWrapper(state, xpts, alen, &tstep);
        }
        else if  (!scheme.compare("S"))
        {
            tfm = sweptWrapper(state, xpts, alen, &tstep);
        }
        else
        {
            std::cerr << "Incorrect or no scheme given" << std::endl;
        }


        MPI_Barrier(MPI_COMM_WORLD);
        if (!ranks[1]) timed = (MPI_Wtime() - timed);
        if (cGlob.hasGpu)  cudaDeviceSynchronize();

        for (int i=0; i<nrows; i++)
        {
            for (int k=1; k<alen[i]; k++)  solutionOutput(state[i], tfm, k, xpts[i]);
        }

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        if (!ranks[1])
        {
            timed *= 1.e6;

            double n_timesteps = tfm/cGlob.dt;

            double per_ts = timed/n_timesteps;

            std::cout << n_timesteps << " timesteps" << std::endl;
            std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

            // Write out performance data as csv
            std::string tpath = pth + "/t" + fspec + scheme + t_ext;
            FILE * timeOut;
            timeOut = fopen(tpath.c_str(), "a+");
            if (timeOut==NULL) fprintf(timeOut, "tpb,gpuA,nX,time\n");
            fprintf(timeOut, "%d,%.4f,%d,%.8f\n", cGlob.tpb, cGlob.gpuA, cGlob.nX, per_ts);
            fclose(timeOut);
        }
    }
        //WRITE OUT JSON solution to differential equation

	#ifndef NOS
        std::string spath = pth + "/s" + fspec + "_" + myrank + i_ext;
        cout << spath << endl;
        std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
        if (!ranks[1]) solution["meta"] = inJ;
        soljson << solution;
        soljson.close();
	#endif

    if (cGlob.hasGpu)
    {
        for (int k=0; k<3; k++) cudaFreeHost(state[k]);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    else
    {
        delete[] state[0];
    }
    delete[] state;

    endMPI();
    return 0;
}

//inline void cudaCheck(cudaError_t code, const char *file, int line, bool abort=false)
//{
//   if (code != cudaSuccess)
//   {
//      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
//      if (abort) exit(code);
//   }
//}
