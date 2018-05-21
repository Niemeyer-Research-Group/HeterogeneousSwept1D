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

writeTime(double tMeas, double tSim, std::string tpath)
{

    tMeas *= 1.e6;

    double n_timesteps = tSim/cGlob.dt;
    double per_ts = tMeas/n_timesteps;

    std::cout << n_timesteps << " timesteps" << std::endl;
    std::cout << "Averaged " << per_ts << " microseconds (us) per timestep" << std::endl;

    // Write out performance data as csv
    FILE * timeOut;
    timeOut = fopen(tpath.c_str(), "a+");
    fseek(timeOut, 0, SEEK_END);
    int ft = ftell(timeOut);
    if (!ft) fprintf(timeOut, "tpb,gpuA,nX,time\n");
    fprintf(timeOut, "%d,%.4f,%d,%.8f\n", cGlob.tpb, cGlob.gpuA, cGlob.nPts, per_ts);
    fclose(timeOut);

}
int main(int argc, char *argv[])
{
    makeMPI(argc, argv);

    if (!rank) cudaRunCheck();

    #ifdef NOS
        if (!rank) std::cout << "No Solution Version." << std::endl;
    #endif

    std::string t_ext = ".csv";
    std::string scheme = argv[1];

    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();

    std::vector<Region *> regions;
   
    setRegion(regions);

    regions.shrink_to_fit();
    std::string pth = argv[3];

    for (auto r: regions)
    {
        r->initializeState(scheme, pth);
    }

    // If you have selected scheme I, it will only initialize and output the initial values.
    if (scheme.compare("I"))
    {
        double timed, tfm;
		if (!rank)
		{
            printf ("Scheme: %s - Grid Size: %d - Affinity: %.2f\n", scheme.c_str(), cGlob.nPts, cGlob.gpuA);
            printf ("threads/blk: %d - timesteps: %.2f\n", cGlob.tpb, cGlob.tf/cGlob.dt);
		}

        MPI_Barrier(MPI_COMM_WORLD);
        if (!rank) timed = MPI_Wtime();

        if (!scheme.compare("C"))
        {
            tfm = classicWrapper(regions);
        }
        else if  (!scheme.compare("S"))
        {
            tfm = sweptWrapper(regions);
        }
        else
        {
            std::cerr << "Incorrect or no scheme given" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (!rank) timed = (MPI_Wtime() - timed);

        if (cGlob.hasGpu)  
		{
			cudaError_t error = cudaGetLastError();
        	if(error != cudaSuccess)
        	{
            	// print the CUDA error message and exit
            	printf("CUDA error: %s\n", cudaGetErrorString(error));
            	exit(-1);
        	}
        }
        if (!rank) writeTime(timed, tfm, pth + "/t" + fspec + scheme + t_ext);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);    

    for (int i=0; i<regions.size(); i++)
    {   
        delete regions[i];        
    }
    regions.clear();

    endMPI();

    return 0;
}

