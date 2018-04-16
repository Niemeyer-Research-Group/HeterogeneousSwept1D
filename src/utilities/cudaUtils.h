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
#include <mpi.h>

#include <cmath>
#include <vector>
#include "myVectorTypes.h"

void cudaRunCheck()
{
    int rv, dv;
    cudaDriverGetVersion(&dv);
    cudaRuntimeGetVersion(&rv);
    printf("CUDA Driver Version / Runtime Version  --- %d.%d / %d.%d\n", dv/1000, (dv%100)/10, rv/1000, (rv%100)/10);
}

int* factor(int n)
{
    int sq = std::sqrt(n);
    int outf = n/sq;
    while (outf * sq != n)
    {
        sq--;
        outf=n/sq;
    }
    static int factors[2] = {sq, outf};
    return factors;
}

struct cudaTime
{
    std::vector<float> times;
    cudaEvent_t start, stop;
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

    void tfinal() { 
        cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime( &ti, start, stop);
        ti *= 1.0e3;
        times.push_back(ti); 
    };

    float getLastTime()
    {
        return ti;
    };

    float avgt() { 
        double sumt = 0.0;
        for (auto itime: times) sumt += itime;
        return sumt/(double)times.size();
    };
};

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