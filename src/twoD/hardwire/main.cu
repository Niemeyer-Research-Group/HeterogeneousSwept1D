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