/**
    DECOMP CORE And other things.
*/

#include <numeric>
#include "gpuDetector.h"

/*
    Globals needed to execute simulation.  Nothing here is specific to an individual equation
*/

void endMPI();

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
    int regionSide, regionBase;
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
    Neighbor(Address addr, short sidx): id(addr), sidx(sidx), ridx((sidx + 2) % 4) , sameProc(rank == id.owner) {}
};

typedef std::array <Neighbor *, 4> stencil;

struct Region
{    
    const Address self;
    const stencil neighbors;
    int xsw, ysw; //Needs the size information.
    int regionAlloc, rows, cols;
    int tstep;
    std::string xstr, ystr, tstr, spath, scheme;
    
    jsons solution; 

    states *state;
    states **stateRows; // Convenience accessor.

    
    Region(Address stencil[5]) 
    : self(stencil[0]), neighbors(meetNeighbors(&stencil[1]))
    {
        xsw = cGlob.regionSide * self.globalx;
        ysw = cGlob.regionSide * self.globaly;
        tstep = TSTEPI;
    }

    ~Region()
    {
        std::cout << "CLOSE REGION" << std::endl;

        #ifndef NOS
            writeSolution();
        #endif

        cudaFreeHost(state);
        delete [] stateRows;
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

    void initializeState(std::string algo, std::string pth)
    {
        spath = pth + "/s" + fspec + "_" + std::to_string(rank) + ".json";
        scheme = algo;
        int exSpace = (!scheme.compare("S")) ? cGlob.ht : 2;
        rows = (cGlob.yPointsRegion + exSpace);
        cols = (cGlob.xPointsRegion + exSpace);
        regionAlloc =  rows * cols ;

        stateRows = new states*[rows];
    

        cudaHostAlloc((void **) &state, regionAlloc*cGlob.szState, cudaHostAllocDefault);
  
        for (int j=0; j<rows; j++)
        {
            stateRows[j] = (states *) state+(j*cols); 
        }
        for(int k=0; k<cGlob.yPointsRegion; k++)
        {   
            for (int i=0; i<cGlob.xPointsRegion; i++)
            {   
                initState(&stateRows[k][i], i+xsw, k+ysw);
            }
        }
        solutionOutput(0.0);
    }

    // O for out 1 for in
    //Ok but how do I use this for MPI transfers?
    void gpuCopy(states *dState, int dir, cudaStream_t stream)
    {
        if (dir)
        {
            cudaMemcpyAsync(state, dState, cudaMemcpyDeviceToHost, st1);
        }
        else
        {
            cudaMemcpyAsync(state, dState, cudaMemcpyHostToDevice, st1);
        }
    }

    void solutionOutput(double tstamp)
    {
        tstr = std::to_string(tstamp);
        for(int k=0; k<cGlob.yPointsRegion; k++)
        {   
            ystr = std::to_string(cGlob.dy * (k+ysw));

            for (int i=0; i<cGlob.xPointsRegion; i++)
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
        std::cout << spath << std::endl;
        if (!rank) solution["meta"] = inJ;
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
    std::cout << " -- MADE IT -- 2.0" << std::endl;
    int localRegions = 1 + cGlob.hasGpu*(cGlob.gpuA - 1);
    int regionMap[nprocs];
    MPI_Allgather(&localRegions, 1, MPI_INT, &regionMap[0], 1, MPI_INT, MPI_COMM_WORLD);
    int tregion = 0;
    Address gridLook[cGlob.yRegions][cGlob.xRegions];
    int k, i;
    int xLoc, yLoc;

    std::cout << " -- MADE IT -- 2.0" << std::endl;

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
                std::cout << " -- MADE IT -- 2." << i << std::endl;
            }  
                
        }
    }
}


void initArgs()
{
    int dimFinder[2];

    //VALUES THAT NEED NO INTROSPECTION
    cGlob.shapeRequest = inJ["Shape"].asString(); //How square required?
    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.ly = inJ["ly"].asDouble();
    cGlob.szState = sizeof(states);
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;

    // IF TPB IS NOT DIVISIBLE BY 32 PICK CLOSEST VALUE THAT IS.
    int tpbReq = inJ["tpb"].asInt();
    tpbx = 32;
    tpby = tpbReq/tpbx;

    // FIND NUMBER OF GPUS AVAILABLE.
	cGlob.gpuA = inJ["gpuA"].asInt(); // AFFINITY
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
    factor(cGlob.nRegions, &dimFinder[0]);
    // How Similar should the dimensions be?

    while (dimFinder[1]/dimFinder[0] > 2)
    {
        if (!cGlob.nGpu)
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
    
    // Something that makes this work if you give it nRegions rather than nPts.
    cGlob.nPoints = inJ["nPts"].asInt();

    cGlob.nPoints = cGlob.nPointsBlock * cGlob.nRegions * cGlob.nBlocksRegion;
    cGlob.xBlocksRegion = dimFinder[1];
    cGlob.yBlocksRegion = dimFinder[0];
    cGlob.xPointsRegion = cGlob.xPointsBlock * cGlob.xBlocksRegion;
    cGlob.yPointsRegion = cGlob.xPointsBlock * cGlob.yBlocksRegion;
    cGlob.xPoints = cGlob.xPointsRegion * cGlob.xRegions;
    cGlob.yPoints = cGlob.yPointsRegion * cGlob.yRegions;
    
    inJ["nPts"] = cGlob.nPoints;
    inJ["nX"] = cGlob.xPoints;
    inJ["nY"] = cGlob.yPoints;

    cGlob.ht = (cGlob.xPointsBlock-2)/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;

    // Derived quantities

    cGlob.dx = cGlob.lx/(double)cGlob.xPoints; // Spatial step
    cGlob.dy = cGlob.ly/(double)cGlob.yPoints; // Spatial step
    cGlob.nWrite = cGlob.tf/cGlob.freq + 2;
    inJ["dx"] = cGlob.dx; // To send back to equation folder. 
    inJ["dy"] = cGlob.dy;
    
    HCONST.init(inJ);
    cudaMemcpyToSymbol(DCONST, &HCONST, sizeof(HCONST));
    if (!rank)  std::cout << rank <<  " - Initialized Arguments -" << std::endl;
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
