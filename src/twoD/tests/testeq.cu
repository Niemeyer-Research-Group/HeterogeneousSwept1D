/**
---------------------------
    TEST CASE 1
---------------------------
*/

#include "cudaUtils.h"
#include "../equations/wave.h"
#include "../decomposition/decomp.h"
#include "../decomposition/classic.h"
#include "../decomposition/swept.h"
#include <vector>
#include <cuda.h>
#include <array>

__device__
void printme(double *poi, int k)
{
    printf("%.d, %.2f ", k, poi[k]);
}

__device__ 
void setSMEM(double *smem, double *gmem, int gid, int k)
{
    smem[gid-k] = gmem[gid]+10.5;
}

__global__
void testMidShared(double *globptr)
{
    double *localptr = (double *) (globptr); 
    extern __shared__ double sptr[];
    
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int k = 0;
    bool b = true;
    while (k < (blockDim.x >> 1))
    {
        if (gid>k && gid<(blockDim.x-k)) 
        {
            printme(localptr, k);
            if ((blockDim.x-2*k) < 17 && b) 
            {
                setSMEM(sptr, localptr, gid, k);
                b=false;
            }
        }
        if ((blockDim.x-2*k) < 17) localptr = sptr;

        k++;
        __syncthreads();
    }
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
    if (!rank) std::cout << inJ << std::endl;

    std::vector<Region *> regions;
   
    setRegion(regions);
    regions.shrink_to_fit();
    std::string pth = argv[3];

    for (auto r: regions)
    {
        r->initializeState(scheme, pth);
        // std::cout << (rank & 1) << std::endl;
        // printf("------------ %d, %d -------\n", rank, r->self.localIdx);
        // for (auto n: r->neighbors) n->printer();
    }
   


    int np = 32;
    double hin[np];
    for (int i=0; i<np; i++) hin[i] = ((double) i)+0.2; 

    double *jean;
    int alloc = sizeof(double) * np;
    int salloc = alloc/2;
    cudaMalloc((void **) &jean, alloc);
    cudaMemcpy(jean, hin, alloc, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaFree(jean);

    if (!scheme.compare("S"))
    {
        sweptWrapper(regions);
    }
    else
    {
        classicWrapper(regions);
    }

    MPI_Barrier(MPI_COMM_WORLD);    

    for (int i=0; i<regions.size(); i++)
    {   
        delete regions[i];        
    }
    regions.clear();
    
    endMPI();

    return 0;
}