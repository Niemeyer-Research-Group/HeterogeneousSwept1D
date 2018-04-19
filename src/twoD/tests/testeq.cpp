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

    std::string i_ext = ".json";
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

    // FOR REGION IN REGIONS WRITE IT OUT.
    
    return 0;
}