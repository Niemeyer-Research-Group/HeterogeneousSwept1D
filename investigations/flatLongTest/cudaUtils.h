/*
    UTILITIES FOR HSWEEP.  TIMER and DRIVER AGREEMENT CHECKER.
*/

#ifndef SWEEPUTIL_H
#define  SWEEPUTIL_H
#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>
#include <array>

// template< typename T >
// void check(T result, char const *const func, const char *const file, int const line)
// {
//     if (result)
//     {
//         fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
//                 file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
//         DEVICE_RESET
//         // Make sure we call CUDA Device Reset before exiting
//         exit(EXIT_FAILURE);
//     }
// }

// #define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

void cudaKernelCheck(bool gpuq=true)
{
    if (gpuq)
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
}

// From cppOverload in CUDASamples.
#define OUTPUT_ATTR(attr)  \
    printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);   \
    printf("Constant Size: %d\n", (int)attr.constSizeBytes);                 \
    printf("Local Size:    %d\n", (int)attr.localSizeBytes);                 \
    printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock);          \
    printf("Number of Registers: %d\n", attr.numRegs);                       \
    printf("PTX Version: %d\n", attr.ptxVersion);                            \
    printf("Binary Version: %d\n", attr.binaryVersion);                      \

void cudaRunCheck()
{
    int rv, dv;
    cudaDriverGetVersion(&dv);
    cudaRuntimeGetVersion(&rv);
    printf("CUDA Driver Version / Runtime Version  --- %d.%d / %d.%d\n", dv/1000, (dv%100)/10, rv/1000, (rv%100)/10);
}

void factor(int n, int* factors)
{
    int sq = std::sqrt(n);
    int outf = n/sq;
    while (outf * sq != n)
    {
        sq--;
        outf=n/sq;
    }
    factors[0] = sq;
    factors[1] = outf;
}

struct cudaTime
{
    std::vector<float> times;
    cudaEvent_t start, stop;
    int nTimes=0;
	float ti;
    std::string typ = "GPU";

    cudaTime() {
        cudaEventCreate( &start );
	    cudaEventCreate( &stop );
    };
    ~cudaTime()
    {
        cudaEventDestroy( start );
	    cudaEventDestroy( stop );
    };

    void tinit(){ cudaEventRecord( start, 0); };

    void tfinal(double nTimesteps=1.0) {
        cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime( &ti, start, stop);
        ti *= 1.0e3;
        nTimes++;
        times.push_back(ti/nTimesteps);
    };

    float getLastTime()
    {
        return times.back();
    };

    float avgt() {
        double sumt = 0.0;
        for (auto itime: times) sumt += itime;
        return sumt/(double)times.size();
    };

    friend std::ostream& operator<< (std::ostream &out, const cudaTime &ct);
};


std::ostream& operator<< (std::ostream &out, const cudaTime &ct)
{
    out << "Simulation Time (us) - ";
    for (auto itt: ct.times)
        out << itt << " | ";

    return out;
}

#ifdef MPIS
#include <mpi.h>

struct mpiTime
{
    std::vector<double> times;
    double ti;

    std::string typ = "CPU";

    void tinit(){ ti = MPI_Wtime(); }

    void tfinal() { times.push_back((MPI_Wtime()-ti)*1.e6); }

    int avgt() {
        double sumt = 0.0;
        for (auto itime: times) sumt += itime;
        return sumt/(double)times.size();
    };
};

#endif
// void atomicWrite(std::string st, std::vector<double> t)
// {
//     FILE *tTemp;
//     MPI_Barrier(MPI_COMM_WORLD);

//     for (int k=0; k<nprocs; k++)
//     {
//         if (ranks[1] == k)
//         {
//             tTemp = fopen(fname.c_str(), "a+");
//             fseek(tTemp, 0, SEEK_END);
//             fprintf(tTemp, "\n%d,%s", ranks[1], st.c_str());
//             for (auto i = t.begin(); i != t.end(); ++i)
//             {
//                 fprintf(tTemp, ",%4f", *i);
//             }
//         }
//         MPI_Barrier(MPI_COMM_WORLD);
//     }
// }

#endif