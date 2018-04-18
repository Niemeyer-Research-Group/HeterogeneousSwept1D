/**
    DECOMP CORE And other things.
*/

#include <numeric>
#include "gpuDetector.h"
#include "json/jsons.h"
#include "mpi.h"

typedef Json::Value jsons;

#define TAGS(x) x & 32767

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

__global__ 
void mailboxes(states **row, states *mail, int xLen, int yLen, int nrows)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (gid>=nrows) return;

    int idx = (gid/2) * xLen + ((gid/2) + (gid%2)) * yLen; 
    row[gid] = (states *)(mail + idx);
}

__global__ 
void rowcast(states **row, states *state, int xLen, int nrows)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (gid>=nrows) return;

    int idx = gid * xLen;
    row[gid] = (states *)(state + idx);
}

// MPI process properties
MPI_Datatype struct_type;
int lastproc, nprocs, rank;

struct Globalism {
// Topology
    int nGpu, nPts, nWrite, hasGpu;
    double gpuA;

// Geometry-No second word is in overall grid.
    //How Many Regions Per
    int nRegions, xRegions, yRegions;
    //How many Blocks Per
    int nBlocks, xBlocks, yBlocks;
    int nBlocksRegion, xBlocksRegion, yBlocksRegion; 
    // How many points per
    int nPoints, xPoints, yPoints;
    int nPointsRegion, xPointsRegion, yPointsRegion;
    int nPointsBlock, xPointsBlock;
    int ht, htm, htp;
    int szState;

// Iterator
    double tf, freq, dt, dx, dy, lx, ly;

} cGlob;

struct Address
{
    int owner, gpu, localIdx, globalx, globaly;
};

void interProc(Address *sourceAddr, Address *destAddr, states *inbox, states *outbox)
{

}

void intraProc(Address *sourceAddr, Address *destAddr, states *inbox, states *outbox)
{

}

typedef void (*mailCarrier) (Address *, Address *, states *, states *);

struct Neighbor
{
    const Address id;
    mailCarrier passer;
    Neighbor(Address addr): id(addr)
    {
        passer = (rank == id.owner) ? interProc : intraProc;
    } 
}

struct Region
{    
    const Address self;
    const std::vector<Neighbor> neighbors;
    int xsw, ysw; //Needs the size information.
    int regionAlloc, rows, cols;
    std::string xstr, ystr, tstr, scheme;
    
    MPI_Request req[4];
    MPI_Status stat[4];
    jsons solution; 

    states *state, flatIn, flatOut, dState, dFlatIn, dFlatOut; // pts in region
    states **stateRows, inbox, outbox, dStateRows, dInbox, dOutbox; // rows in region
    
    Region(Address stencil[5]) 
    :self(stencil[0])), neighbors(stencil+1, stencil+4), gpu(cGlob.hasGpu) 
    {
        xsw = cGlob.xPointsRegion * self.globalx;
        ysw = cGlob.yPointsRegion * self.globaly;

    }
    ~Region()
    {
        cudaFree(dState);
        cudaFree(dStateRows);
        cudaFree(dFlatOut);
        cudaFree(dFlatIn);
        cudaFree(dInbox);
        cudaFree(dOutbox);

        cudaFreeHost(state);
    }

    void gpuInit()
    {
        int mailSize = 2*(cGlob.xPointsRegion + cGlob.yPointsRegion)
        cudaMalloc((void**) &dInbox, sizeof(*states)*4);
        cudaMalloc((void**) &dOutbox, sizeof(*states)*4);
        cudaMalloc((void**) &dStateRows, sizeof(*states)*rows);
        
        cudaMalloc((void**) &dFlatIn, sizeof(states)*mailSize);
        cudaMalloc((void**) &dFlatOut, sizeof(states)*mailSize);
        cudaMalloc((void**) &dState, sizeof(states)*regionAlloc);

        mailboxes<<<1, 4>>>(dFlatIn, dInbox, cGlob.xPointsRegion, cGlob.yPointsRegion, 4);
        mailboxes<<<1, 4>>>(dFlatOut, dOutbox, cGlob.xPointsRegion, cGlob.yPointsRegion, 4);
        int tbx = rows/64 + 1;
        rowcast<<<tbx, 64>>>(dStateRows, dState, cols, rows); 

        cudaStream_t streams[cGlob.gpuA];
        for (int i = 0; i < cGlob.gpuA; i++) cudaStreamCreate(&streams[i]);
        cudaMemcpy(dState, state, sizeof(states)*regionAlloc, cudaMemcpyHostToDevice);
    }
    // Proto Caller
    for (int i = 0; i < cGlob.gpuA; i++)
    {
        classicStep <<< blockGrid, threadGrid, 0, streams[i] >>> (Regions stateRow, Regions passer, tstep);
    }

    void initializeState(std::string algo)
    {
        scheme = algo;
        int exSpace = (!scheme.compare("S")) ? cGlob.ht : 2;
        rows = (cGlob.yPointsRegion + exSpace);
        cols = (cGlob.xPointsRegion + exSpace);
        regionAlloc =  rows * cols ;

        stateRows = new states*[rows];
    
        if(gpu)
        {
            cudaHostAlloc((void **) &state, regionAlloc*cGlob.szState, cudaHostAllocDefault);
        }
        else 
        {
            state = (states*) malloc(regionAlloc * cGlob.szState);
        }
        for (int j=0; j<rows; j++)
        {
            stateRows[j] = (states *) state+(j*cols); 
        }
        for(int k=0; k<cGlob.yPointsRegion; k++)
        {   
            for (int i=0; i<cGlob.xPointsRegion; i++)
            {   
                xstr = std::to_string(i*cGlob.dx); 
                initState(&stateRows[k][i], k+ysw, i+xsw);
            }
        }
        if (gpu) gpuInit(cols);
    }

    void solutionOutput(double tstamp)
    {
        tstr = std::to_string(tstamp);
        for(int k=0; k<cGlob.yPointsRegion; k++)
        {   
            ystr = std::to_string(k*cGlob.dy);
            for (int i=0; i<cGlob.xPointsRegion; i++)
            {   
                xstr = std::to_string(i*cGlob.dx); 
                for (int j=0; j<NVARS; j++)
                {
                    solution[outVars[j]][tstr][ystr][xstr] = HCONST.printout(&outState[k][i], j);
                }
            }
        }
        
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
			inJ[inarg] = atof(argv[k+1]);
        }
    }
}

void setRegion(Region **regionals)
{
    int localRegions = 1 + cGlob.hasGpu*(cGlob.gpuA - 1);
    regionals = new Region* [localRegions];
    int regionMap[nProcs];
    MPI_Allgather(&localRegions, 1, MPI_INT, &regionMap[0], 1, MPI_INT, MPI_COMM_WORLD);
    int tregion = 0;
    Address gridLook[cGlob.yRegions][cGlob.xRegions];
    int k, i;
    int xLoc, yLoc;

    for (k = 0; k<nProcs; k++)
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
                }

                regionals[cnt] = new Region(addr, cGlob.hasGpu);
            }
        }
    }
}

void initArgs()
{
    int *dimFinder;

    //VALUES THAT NEED NO INTROSPECTION

    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.ly = inJ["ly"].asDouble();
    cGlob.szState = sizeof(states);
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    // IF TPB IS NOT SQUARE AND DIVISIBLE BY 32 PICK CLOSEST VALUE THAT IS.
    int tpbReq = inJ["tpb"].asInt();
    cGlob.xPointsBlock = std::sqrt(tpbReq);
    if (cGlob.xPointsBlock % 8) cGlob.xPointsBlock = 8 * (1 + (cGlob.xPointsBlock/8)); 
    cGlob.nPointsBlock = cGlob.xPointsBlock * cGlob.xPointsBlock;
    inJ["blockSide"] = cGlob.xPointsBlock;

    // FIND NUMBER OF GPUS AVAILABLE.
	cGlob.gpuA = inJ["gpuA"].asDouble(); // AFFINITY
	int ranker = rank;
	int sz = nprocs;

	if (!cGlob.gpuA)
    {
        cGlob.hasGpu = 0;
        cGlob.nGpu = 0;
    }
    else
    {
        cGlob.hasGpu = detector(ranker, sz, 0);
        MPI_Allreduce(&cGlob.hasGpu, &cGlob.nGpu, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        sz -= cGlob.nGpu;
    }

    cGlob.nRegions = sz + (cGlob.nGpu * cGlob.gpuA);
    dimFinder = factor(cGlob.nRegions);
    
    // WE JUST WANT THE DIMENSIONS TO BE RELATIVELY SIMILAR.
    while (dimFinder[1]/dimFinder[0] > 2)
    {
        if (!cGlob.nGpu)
        {
            std::cout << "You provided a prime number of processes and no GPU capability. That's just silly and you absolutely cannot do it because we must be able to form a rectangle from the number of processes on each device.  Retry!" << std::endl;
            exit();
        }

        cGlob.gpuA++;
        cGlob.nRegions = sz + (cGlob.nGpu * cGlob.gpuA);
        dimFinder = factor(cGlob.nRegions);
    }
    cGlob.xRegions = dimFinder[0]; 
    cGlob.yRegions = dimFinder[1];

    // Something that makes this work if you give it nRegions rather than nPts.
    cGlob.nPoints = inJ["nPts"].asInt();
    cGlob.nBlocksRegion = cGlob.nPoints/(cGlob.nPointsBlock * cGlob.nRegions);
    dimFinder = factor(cGlob.nBlocksRegion);

    while (dimFinder[1]/dimFinder[0] > 2)
    {
        cGlob.nBlocksRegion++;
        dimFinder = factor(cGlob.nBlocksRegion);
    }

    cGlob.nPoints = cGlob.nPointsBlock * cGlob.nRegions * cGlob.nBlocksRegion;
    cGlob.xBlocksRegion = dimFinder[1];
    cGlob.yBlocksRegion = dimFinder[0];
    cGlob.xPointsRegion = cGlob.xPointsBlock * cGlob.xBlocksRegion;
    cGlob.yPointsRegion = cGlob.xPointsBlock * cGlob.yBlocksRegion;
    cGlob.xPoints = cGlob.xPointsRegion * cGlob.xRegions;
    cGlob.yPoints = cGlob.yPointsRegion * cGlob.yRegions;

    cGlob.ht = (cGlob.xPointsBlock-2)/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;
    // Derived quantities

    cGlob.dx = cGlob.lx/(double)cGlob.nx; // Spatial step
    cGlob.dy = cGlob.ly/(double)cGlob.ny; // Spatial step
    cGlob.nWrite = cGlob.tf/cGlob.freq + 2;
    inJ["dx"] = cGlob.dx; // To send back to equation folder. 
    inJ["dy"] = cGlob.dy;

    HCONST.init(inJ);
    cudaMemcpyToSymbol(DCONST, &HCONST, sizeof(HCONST));

    if (!rank)  std::cout << rank <<  " - Initialized Arguments -" << std::endl;
}
struct mpiTime
{
    std::vector<double> times;
    double ti;
    std::string typ = "CPU";

    void tinit(){ ti = MPI_Wtime(); }

    void tfinal() { times.push_back((MPI_Wtime()-ti)*1000.0); }

    int avgt() { 
        return std::accumulate(times.begin(), times.end(), 0)/ (float)times.size();
        }
};

void atomicWrite(std::string st, std::vector<double> t)
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
