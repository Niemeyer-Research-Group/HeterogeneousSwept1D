/*
    Entry point for hsweep.
*/
#include <fstream>

#include "heads.h"
#include "decomp.h"
#include "classic.h"
#include "swept.h"

/**
----------------------
    MAIN PART
----------------------
*/

int main(int argc, char *argv[])
{
    makeMPI(argc, argv);

    if (!ranks[1]) cudaRunCheck();

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

    Region **regions;
    setRegion(regions);

    std::string pth = string(argv[3]);

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
            tfm = classicWrapper(state, &tstep);
        }
        else if  (!scheme.compare("S"))
        {
            tfm = sweptWrapper(state, &tstep);
        }
        else
        {
            std::cerr << "Incorrect or no scheme given" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (!ranks[1]) timed = (MPI_Wtime() - timed);
        if (cGlob.hasGpu)  
		{
			cudaError_t error = cudaGetLastError();
        	if(error != cudaSuccess)
        	{
            	// print the CUDA error message and exit
            	printf("CUDA error: %s\n", cudaGetErrorString(error));
            	exit(-1);
        	}
			cudaDeviceSynchronize();
		}

        writeOut(state, tfm);

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
            fseek(timeOut, 0, SEEK_END);
            int ft = ftell(timeOut);
            if (!ft) fprintf(timeOut, "tpb,gpuA,nX,time\n");
            fprintf(timeOut, "%d,%.4f,%d,%.8f\n", cGlob.tpb, cGlob.gpuA, cGlob.nX, per_ts);
            fclose(timeOut);
        }
    }
        //WRITE OUT JSON solution to differential equation

	#ifndef NOS
        std::string spath = pth + "/s" + fspec + "_" + myrank + i_ext;
        std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
        if (!ranks[1]) solution["meta"] = inJ;
        soljson << solution;
        soljson.close();
	#endif

    if (cGlob.hasGpu)
    {
        cudaFreeHost(state);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    else
    {
        free(state);
    }

    endMPI();
    return 0;
}

