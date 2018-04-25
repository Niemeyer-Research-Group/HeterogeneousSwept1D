/**
    DECOMP CORE And other things.
*/

#include <numeric>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpuDetector.h"

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

void endMPI();

__global__
void setgpuRegionA(states *rdouble, states *rmember)
{
    printf("ONCE!\n");
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (gid>1) return;

    printf("ONCE! %.f", rmember[gid]);
    rdouble = (states *)(&rmember[0]);
}

// MPI process properties
MPI_Datatype struct_type;
int lastproc, nprocs, rank;

struct Globalism
{
    // Topology || Affinity is int now.
    int nGpu, nWrite, hasGpu, procRegions, gpuA;
    std::string shapeRequest;

    // Geometry-No second word is in overall grid.
    //How Many Regions Per
    int tpbx, tpby;
    int nRegions, xRegions, yRegions;
    // How many points per
    int nPoints, xPoints, yPoints;
    int regionPoints, regionSide, regionBase;
    int ht, htm, htp;
    int szState;

    // Iterator
    double tf, freq, dt, dx, dy, lx, ly;

} cGlob;

struct Address
{
    int owner, gpu, localIdx, globalx, globaly;
};

// typedef void (*mailCarrier) (Address *, Address *, states *, states *);

struct Neighbor
{
    const Address id;
    const short sidx, ridx;
    const bool sameProc;
    MPI_Request req;
    MPI_Status stat;
    Neighbor(Address addr, short sidx): id(addr), sidx(sidx), 
    ridx((sidx + 2) % 4) , sameProc(rank == id.owner) {}
};

typedef std::array <Neighbor *, 4> stencil;
jsons solution; 
struct Region
{    
    const Address self;
    const stencil neighbors;
    int xsw, ysw;
    int regionAlloc, fullSide, copyBytes;
    int tStep, stepsPerCycle;
    double tStamp, tWrite;
    std::string xstr, ystr, tstr, spath, scheme;
    
    states *state;
    states **stateRows; // Convenience accessor.
    // GPU STUFF
    states *dState;
    cudaStream_t stream;

    Region(Address stencil[5]) 
    : self(stencil[0]), neighbors(meetNeighbors(&stencil[1])), tStep(0), tStamp(0.0), tWrite(cGlob.freq)
    {
        xsw = cGlob.regionSide * self.globalx;
        ysw = cGlob.regionSide * self.globaly;
    }

    ~Region()
    {
        #ifndef NOS
            if(self.gpu) gpuCopy(true);
            solutionOutput();
            writeSolution();
        #endif

        if (self.gpu)
        {
            cudaStreamDestroy(stream);
            cudaFree(dState);
        }
        
        delete[] stateRows;
        delete[] state;
        
        for (int i=0; i<3; i++)  delete neighbors[i];
    }

    stencil meetNeighbors(Address *addr)
    {
        stencil n;
        for(int k=0; k<4; k++)
        {
            n[k] = new Neighbor(addr[k], k);
        }
        return n;
    }

    void incrementTime()
    {
        tStep += stepsPerCycle;
        tStamp = (double)tStep * cGlob.dt; 
        if (tStamp > tWrite)
        {
            if (self.gpu) gpuCopy(true);
            solutionOutput();
        } 
    }

    void makeGPUState()
    {   
        cudaStreamCreate(&stream);
        cudaMalloc((void **) &dState, regionAlloc);
        gpuCopy(false);
    }

    void setGPUPointers(states *dDouble)
    {
        std::cout << "PORQUE NO" << std::endl;
        setgpuRegionA <<< 1, 1 >>> (dDouble, dState);
        // cudaDeviceSynchronize();
    }

    // True is Down to Host.  False is Up to Device.
    void gpuCopy(int dir)
    {   
        if (!self.gpu) return; //Guards against unauthorized access. 

        if (dir)
        {
            cudaMemcpyAsync(state, dState, regionAlloc, cudaMemcpyDeviceToHost, stream);
        }
        else
        {
            cudaMemcpyAsync(dState, state, regionAlloc, cudaMemcpyHostToDevice, stream);
        }
    }

    void initializeState(std::string algo, std::string pth)
    {
        spath = pth + "/s" + fspec + "_" + std::to_string(rank) + ".json";
        scheme = algo;

        int exSpace;
        if (!scheme.compare("S")) 
        {
            exSpace         = cGlob.htp;
            stepsPerCycle   = cGlob.ht;
        }
        else
        {
            exSpace         = 2;
            stepsPerCycle   = 1;
        }

        fullSide    = (cGlob.regionSide + exSpace);
        regionAlloc = fullSide * fullSide * cGlob.szState;
        copyBytes   = cGlob.regionPoints * cGlob.szState;
        
        stateRows   = new states*[fullSide];
        state       = new states[fullSide*fullSide];

        if (self.gpu)   makeGPUState();       

        for (int j=0; j<fullSide; j++)
        {
            stateRows[j] = (states *) state+(j*fullSide); 
        }
        for(int k=0; k<cGlob.regionSide+2; k++)
        {   
            for (int i=0; i<cGlob.regionSide+2; i++)
            {   
                initState(&stateRows[k][i], i+xsw-1, k+ysw-1);
            }
        }

        solutionOutput();
        
        tStep   += 2;
        tStamp  = (double)tStep * cGlob.dt;
    }

    void solutionOutput()
    {
        tstr = std::to_string(tStamp);
     
        for(int k=0; k<cGlob.regionSide; k++)
        {   
            ystr = std::to_string(cGlob.dy * (k+ysw));

            for (int i=0; i<cGlob.regionSide; i++)
            {   
                xstr = std::to_string(cGlob.dx * (i+xsw)); 

                for (int j=0; j<NVARS; j++)
                {
                    solution[outVars[j]][tstr][ystr][xstr] = printout(&stateRows[k][i], j);
                }
            }
        }
    }

    void writeSolution()
    {
        std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
        //if (!rank) solution["meta"] = inJ;
        soljson << solution;
        soljson.close();
    }
};

jsons inJ;

void makeMPI(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    mpi_type(&struct_type);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lastproc = nprocs-1;
}

// I think this needs a try except for string inputs.
void parseArgs(int argc, char *argv[])
{
    if (argc>4)
    {
        std::string inarg;
        for (int k=4; k<argc; k+=2)
        {
            inarg = argv[k];
			inJ[inarg] = argv[k+1];
        }
    }
}

void setRegion(std::vector <Region *> &regionals)
{
    int localRegions = 1 + cGlob.hasGpu*(cGlob.gpuA - 1);
    int regionMap[nprocs];
    MPI_Allgather(&localRegions, 1, MPI_INT, &regionMap[0], 1, MPI_INT, MPI_COMM_WORLD);
    int tregion = 0;
    Address gridLook[cGlob.yRegions][cGlob.xRegions];
    int k, i;
    int xLoc, yLoc;

    for (k = 0; k<nprocs; k++)
    {
        for (i = 0; i<regionMap[k]; i++)
        {
            xLoc = tregion%cGlob.xRegions;
            yLoc = tregion/cGlob.xRegions;

            gridLook[yLoc][xLoc] = {k, cGlob.hasGpu, tregion, xLoc, yLoc};
            tregion++;
        }
    }

    Address addr[5];
    for (k = 0; k<cGlob.yRegions; k++)
    {
        for (i = 0; i<cGlob.xRegions; i++)
        {
            if (gridLook[k][i].owner == rank)
            {
                addr[0] = gridLook[k][i];
                addr[1] = gridLook[(k-1)%cGlob.yRegions][i];
                addr[2] = gridLook[k][(i-1)%cGlob.xRegions];
                addr[3] = gridLook[(k+1)%cGlob.yRegions][i];
                addr[4] = gridLook[k][(i-1)%cGlob.xRegions];
                regionals.push_back(new Region(addr));
            }  
                
        }
    }
}

double tolerance(std::string shape)
{
    if      (shape == "Perfect")    return 1.05;
    else if (shape == "Strict")     return 1.5;
    else if (shape == "Moderate")   return 4.0;
    else if (shape == "Loose")      return 20.0;
    else                            return 1.5;
}

void initArgs()
{
    int dimFinder[2];

    //VALUES THAT NEED NO INTROSPECTION
    cGlob.dt        = inJ["dt"].asDouble();
    cGlob.tf        = inJ["tf"].asDouble();
    cGlob.freq      = inJ["freq"].asDouble();
    cGlob.lx        = inJ["lx"].asDouble();
    cGlob.ly        = inJ["ly"].asDouble();
    cGlob.szState   = sizeof(states);
    cGlob.nPoints   = inJ["nPts"].asInt();

    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    // IF TPB IS NOT DIVISIBLE BY 32 PICK CLOSEST VALUE THAT IS.
    int tpbReq = inJ["tpb"].asInt();    
    cGlob.tpbx = 32;
    cGlob.tpby = tpbReq/cGlob.tpbx;

    // FIND NUMBER OF GPUS AVAILABLE.
	cGlob.gpuA  = inJ["gpuA"].asInt(); // AFFINITY
	int ranker  = rank;
	int sz      = nprocs;

	if (!cGlob.gpuA)
    {
        cGlob.hasGpu = 0;
        cGlob.nGpu   = 0;
    }
    else
    {
        cGlob.hasGpu = detector(ranker, sz, 0);
        MPI_Allreduce(&cGlob.hasGpu, &cGlob.nGpu, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        sz -= cGlob.nGpu;
    }
    
    cGlob.nRegions = sz + (cGlob.nGpu * cGlob.gpuA);
    factor(cGlob.nRegions, &dimFinder[0]);
    double tol = tolerance(inJ["Shape"].asString());
    // How Similar should the dimensions be?

    while (((double)dimFinder[1])/dimFinder[0] > tol)
    {
        if (!cGlob.nGpu && dimFinder[0] == 1)
        {
            std::cout << "You provided a prime number of processes and no GPU capability. That's just silly and you absolutely cannot do it because we must be able to form a rectangle from the number of processes on each device.  Retry!" << std::endl;
            endMPI();
            exit(1);
        }

        cGlob.gpuA++;
        cGlob.nRegions = sz + (cGlob.nGpu * cGlob.gpuA);
        factor(cGlob.nRegions, &dimFinder[0]);
    }
    
    cGlob.xRegions = dimFinder[0]; 
    cGlob.yRegions = dimFinder[1];
    
    // Regions must be square because they are the units the pyramids and bridges are built on.
    cGlob.regionPoints  = cGlob.nPoints/cGlob.nRegions;
    cGlob.regionSide    = std::sqrt(cGlob.regionPoints);
    cGlob.regionSide    = 32 * (cGlob.regionSide / 32 + 1); 
    cGlob.regionPoints  = cGlob.regionSide * cGlob.regionSide;

    cGlob.nPoints = cGlob.regionPoints * cGlob.nRegions;
    cGlob.xPoints = cGlob.regionSide * cGlob.xRegions;
    cGlob.yPoints = cGlob.regionSide * cGlob.yRegions;
    
    inJ["nPts"] = cGlob.nPoints;
    inJ["nX"]   = cGlob.xPoints;
    inJ["nY"]   = cGlob.yPoints;

    cGlob.ht    = (cGlob.regionSide-2)/2;
    cGlob.htm   = cGlob.ht-1;
    cGlob.htp   = cGlob.ht+1;

    // Derived quantities

    cGlob.dx        = ((double) cGlob.lx)/cGlob.xPoints; // Spatial step
    cGlob.dy        = ((double) cGlob.ly)/cGlob.yPoints; // Spatial step
    cGlob.nWrite    = cGlob.tf/cGlob.freq + 2;
    inJ["dx"]       = cGlob.dx; // To send back to equation folder. 
    inJ["dy"]       = cGlob.dy;
    
    HCONST.init(inJ);
    cudaMemcpyToSymbol(DCONST, &HCONST, sizeof(HCONST));
    if (!rank)  std::cout << rank <<  " - Initialized Arguments - " << cGlob.nRegions << std::endl;
}

void atomicWrite(std::string st, std::vector<double> t, std::string fname)
{
    FILE *tTemp;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int k=0; k<nprocs; k++)
    {
        if (rank == k)
        {
            tTemp = fopen(fname.c_str(), "a+");
            fseek(tTemp, 0, SEEK_END);
            fprintf(tTemp, "\n%d,%s", rank, st.c_str());
            for (auto i = t.begin(); i != t.end(); ++i)
            {
                fprintf(tTemp, ",%4f", *i);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void endMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&struct_type);
	MPI_Finalize();
}
