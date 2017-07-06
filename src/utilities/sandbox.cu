/*
    Just a testing main program for the functions
*/

// Compile IT:  nvcc -o sandbx sandbox.cu -lm -restrict -gencode arch=compute_35,code=sm_35

#include <iostream>
#include "../equations/Euler/EulerGlobals.h"

using namespace std;

__global__ void assign_kernel(void)
{
    dFunc[0] = pressureRatio1;
    dFunc[1] = eulerHalfStep;
    dFunc[2] = pressureRatio2;
    dFunc[3] = eulerFullStep;
}

void assign(void)
{
    hFunc[0] = pressureRatio1;
    hFunc[1] = eulerHalfStep;
    hFunc[2] = pressureRatio2;
    hFunc[3] = eulerFullStep;
}


int main()
{
    REALthree IC = {2.0,5.0,6.0};
    assign();
    REAL pman = pressure(IC);
    cout << pman << endl;
    return 0;

}