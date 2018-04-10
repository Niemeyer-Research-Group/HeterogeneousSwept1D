/*
    UTILITIES FOR HSWEEP.  TIMER and DRIVER AGREEMENT CHECKER.
*/


#ifdef SWEEPUTIL_H
#define  SWEEPUTIL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

void cudaRunCheck()
{
    int rv, dv;
    cudaDriverGetVersion(&dv);
    cudaRuntimeGetVersion(&rv);
    printf("CUDA Driver Version / Runtime Version  --- %d.%d / %d.%d\n", dv/1000, (dv%100)/10, rv/1000, (rv%100)/10);
}

int factor(int n)
{
    int sq = std::sqrt(n);
    if ((sq * sq) == n) return sq;

    outf = 0;
    for(int k=2; k<sq; k++)
    {
        if (!(n%k)) outf = k;
    }
    return outf;
}

struct cudaTime
{
    std::vector<double> times;
    cudaEvent_t start, stop;
	double ti;
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

    void getLastTime()
    {
        return ti;
    };

    int avgt() { 
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

void atomicWrite(std::string st, std::vector<double> t)
{
    FILE *tTemp;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int k=0; k<nprocs; k++)
    {
        if (ranks[1] == k)
        {
            tTemp = fopen(fname.c_str(), "a+");
            fseek(tTemp, 0, SEEK_END);
            fprintf(tTemp, "\n%d,%s", ranks[1], st.c_str());
            for (auto i = t.begin(); i != t.end(); ++i)
            {
                fprintf(tTemp, ",%4f", *i);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#endif