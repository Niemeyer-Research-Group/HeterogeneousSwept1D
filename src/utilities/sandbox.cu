/*
    Just a testing main program for the functions
*/

// Compile IT:  nvcc -o sandbx sandbox.cu -lm -restrict -gencode arch=compute_35,code=sm_35

//nvcc -o sandbx sandbox.cu ../equations/Euler/Euler_Device.cu -lm -rdc=true -gencode arch=compute_35,code=sm_35 

#include <iostream>
#include <cuda_runtime.h>
#include "EulerGlobals.h"

using namespace std;

int main()
{
    cudaSetDevice(0);
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    hDimens.dt_dx = .0001; // dt/dx
    hDimens.gam = 1.4;
    hDimens.mgam = 0.4;

    cudaMemcpyToSymbol(dDimens, &hDimens,sizeof(dimensions));

    REALthree IC = {2.0, 5.0, 6.0};
    REAL pman = pressure(IC);
    cout << pman << endl;
    return 0;

}