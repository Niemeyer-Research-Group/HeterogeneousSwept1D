/**
---------------------------
    TEST CASE 1
---------------------------
*/

#include "cudaUtils.h"
#include "../equations/wave.h"
#include "../decomposition/decomp.h"
#include <vector>
#include <cuda.h>
#include <array>

__global__
void setgpuRegion(states **rdouble, states *rmember, int i)
{
    printf("ONCE!\n");
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (gid>1) return;

    printf("ONCE! %.f\n", rmember[gid]);
    rdouble[i] = (states *)(&rmember[0]);
}

__global__
void printRegion(states **rdouble)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    states *regional = rdouble[gid];
    const int mid = A.regionSide * A.regionSide;
    printf("gid %.d MiddleItem %.8e\n", gid, regional[mid]);

}

// Classic Discretization wrapper.
void classicWrapper(std::vector <Region *> &regionals)
{
    const int gpuRegions = cGlob.hasGpu * regionals.size();
    states ** regionSelector;
    std::cout << "BEFORE THING - " << rank << std::endl;
    if (gpuRegions) 
    {
        cudaMalloc((void **) &regionSelector, sizeof(states *) * gpuRegions);
        for (int i=0; i<gpuRegions; i++)
        {
            setgpuRegion <<< 1, 1 >>> (regionSelector, regionals[i]->dState, i);
            std::cout << regionals[i]->self.globalx << regionals[i]->self.globaly << std::endl;
        }
        printRegion <<< gpuRegions, 1 >>> (regionSelector);
        cudaFree(regionSelector);
    }
    std::cout << "AFTER THING - " << rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}



int main(int argc, char *argv[])
{
    makeMPI(argc, argv);

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
   
    classicWrapper(regions);
 
    for (int i=0; i<regions.size(); i++)
    {   
        delete regions[i];        
    }
    regions.clear();

    for (auto const& id : solution["Velocity"].getMemberNames()) 
    {
        std::cout << id << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // regions.clear();
    
    endMPI();

    return 0;
}