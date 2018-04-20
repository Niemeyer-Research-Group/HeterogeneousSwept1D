/**
---------------------------
    TEST CASE 1
---------------------------
*/

#include "cudaUtils.h"
#include "../equations/wave.h"
#include "../decomposition/decomp.h"

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

    states testState[2];
    initState(&testState[0], 20, 60);
    std::cout << testState[1].u[0] << std::endl;
    

    std::vector<Region *> regions;
   
    setRegion(regions);


    std::string pth = argv[3];

    for (auto r: regions)
    {
        r->initializeState(scheme, pth);
        r->writeSolution();
    }

    
    // FOR REGION IN REGIONS WRITE IT OUT.
    endMPI();

    return 0;
}