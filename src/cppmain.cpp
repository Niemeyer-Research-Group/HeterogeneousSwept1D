// ALLONE

// Well we definitely need to get rid of the xpts.  Really I need to concentrate on getting the output right so I can check the answers.  Then, if they're right, we can worry about streamlining this. Partly main problem, the keys in the output json are strings. Could read each in and then make it a data frame from dict.

/**
    The equation specific functions.
*/

/**
	The equations specific global variables and function prototypes.
*/

#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime> 
#include <string>
#include <vector>
#include <algorithm>

#include "myVectorTypes.h"
#include "json/json.h"

// We're just going to assume doubles
#define REAL            double
#define REALthree       double3
#define MPI_R           MPI_DOUBLE
#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0
#define SQUAREROOT(x)   sqrt(x)

#define NSTEPS              4
#define NVARS               4

// Since anyone would need to write a header and functions file, why not just hardwire this.  
// If the user's number of steps isn't a power of 2 use the other one.

#define MODULA(x)           x & (NSTEPS-1)
// #define MODULA(x)           x % NSTEPS

#define DIVMOD(x)           (MODULA(x)) >> 1
/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

//---------------// 
struct eqConsts {
    REAL gamma; // Heat capacity ratio
    REAL mgamma; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
};

//---------------// 
struct states {
    REALthree Q[2]; // Full Step, Midpoint step state variables
    REAL Pr; // Pressure ratio
};

std::string outVars[NVARS] = {"DENSITY", "VELOCITY", "ENERGY", "PRESSURE"}; //---------------// 
std::string fspec = "Euler";

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

eqConsts heqConsts; //---------------// 
REALthree hBounds[2]; // Boundary Conditions

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

typedef Json::Value jsons;

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.

    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
#ifdef __CUDA_ARCH__
    #define DIMS    deqConsts
    #define QNAN(x) isnan(x)
    #define QMIN(x, y) min(x, y)
#else
    #define DIMS    heqConsts
    #define QNAN(x) std::isnan(x)
    #define QMIN(x, y) std::min(x, y)
#endif

 double indexer(double dx, int i, int x)
{
    double pt = i+x;
    double dx2 = dx*0.5;
    return dx*pt - dx2;
}

 REAL density(REALthree subj)
{
    return subj.x;
}

 REAL velocity(REALthree subj)
{
    return subj.y/subj.x;
}

 REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

  
__forceinline__
REAL pressure(REALthree qH)
{
    return DIMS.mgamma * (qH.z - (HALF * qH.y * qH.y/qH.x));
}

 REAL printout(states *state, int i)
{
    REALthree subj = state->Q[0];
    REAL ret;

    if (i == 0) ret = density(subj);
    if (i == 1) ret = velocity(subj);
    if (i == 2) ret = energy(subj);
    if (i == 3) ret = pressure(subj);

    return ret;
}

/*
dimensions heqConsts; //---------------// 
REALthree hBound[2]; // Boundary Conditions
double lx; // Length of domain.
*/

 void equationSpecificArgs(jsons inJs)
{
    heqConsts.gamma = inJs["gamma"].asDouble();
    heqConsts.mgamma = heqConsts.gamma - 1;
    REAL rhoL = inJs["rhoL"].asDouble();
    REAL vL = inJs["vL"].asDouble();
    REAL pL = inJs["pL"].asDouble();
    REAL rhoR = inJs["rhoR"].asDouble();
    REAL vR = inJs["vR"].asDouble();
    REAL pR = inJs["pR"].asDouble();
    hBounds[0].x = rhoL;
    hBounds[0].y = vL*rhoL;
    hBounds[0].z = pL/heqConsts.mgamma + HALF * rhoL * vL * vL;
    hBounds[1].x = rhoR;
    hBounds[1].y = vR*rhoR,
    hBounds[1].z = pR/heqConsts.mgamma + HALF * rhoR * vR * vR;
    REAL dtx = inJs["dt"].asDouble();
    REAL dxx = inJs["dx"].asDouble();
    heqConsts.dt_dx = dtx/dxx;
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
 void initialState(jsons inJs, states *inl, int idx, int xst)
{
    double dxx = inJs["dx"].asDouble();
    double xss = indexer(dxx, idx, xst);
    double lx = inJs["lx"].asDouble();
    bool wh = inJs["IC"].asString() == "PARTITION";
    int side;
    if (wh)
    {
        side = (xss > HALF*lx);
        (inl+idx)->Q[0] = hBounds[side];
        (inl+idx)->Q[1] = hBounds[side];
        (inl+idx)->Pr = 0.0;
    }
}

/*
    // MARK : Equation procedure
*/

__forceinline__
REAL pressureRoe(REALthree qH)
{
    return DIMS.mgamma * (qH.z - HALF * qH.y * qH.y);
}

/**
    Ratio
*/
  
__forceinline__
void pressureRatio(states *state, int idx, int tstep)
{
    state[idx].Pr = (pressure(state[idx+1].Q[tstep]) - pressure(state[idx].Q[tstep]))/(pressure(state[idx].Q[tstep]) - pressure(state[idx-1].Q[tstep]));
}   

/**
    Reconstructs the state variables if the pressure ratio is finite and positive.

    @param cvCurrent  The state variables at the point in question.
    @param cvOther  The neighboring spatial point state variables.
    @param pRatio  The pressure ratio Pr-Pc/(Pc-Pl).
    @return The reconstructed value at the current side of the interface.
*/
  
__forceinline__
REALthree limitor(REALthree qH, REALthree qN, REAL pRatio)
{   
    if(QNAN(pRatio) || pRatio<0)  
    {
        return qH;
    }
    else
    {
       return (qH + HALF * QMIN(pRatio, ONE) * (qN - qH));
    }
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
  
__forceinline__ 
REALthree eulerFlux(REALthree qL, REALthree qR)
{
    REAL uLeft = qL.y/qL.x;
    REAL uRight = qR.y/qR.x;

    REAL pL = pressure(qL);
    REAL pR = pressure(qR);

    REALthree flux;
    flux.x = (qL.y + qR.y);
    flux.y = (qL.y*uLeft + qR.y*uRight + pL + pR);
    flux.z = (qL.z*uLeft + qR.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}

/**
    Finds the spectral radius and applies it to the interface.

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The spectral radius multiplied by the difference of the reconstructed values
*/
  
__forceinline__ 
REALthree eulerSpectral(REALthree qL, REALthree qR)
{
    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(qL.x);
    REAL rhoRightsqrt = SQUAREROOT(qR.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*qR.y + rhoRightsqrt*qL.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*qR.z + rhoRightsqrt*qL.z)*halfDenom;

    REAL pH = pressureRoe(halfState);

    return (SQUAREROOT(pH * DIMS.gamma) + fabs(halfState.y)) * (qL - qR);
}

  
void eulerStep(states *state, int idx, int tstep)
{
    REALthree tempStateLeft, tempStateRight;
    int itx = (tstep ^ 1);

    tempStateLeft = limitor(state[idx-1].Q[itx], state[idx].Q[itx], state[idx-1].Pr);
    tempStateRight = limitor(state[idx].Q[itx], state[idx-1].Q[itx], ONE/state[idx].Pr);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Q[itx], state[idx+1].Q[itx], state[idx].Pr);
    tempStateRight = limitor(state[idx+1].Q[itx], state[idx].Q[itx], ONE/state[idx+1].Pr);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    state[idx].Q[tstep] = state[idx].Q[0] + ((QUARTER * (itx+1)) * DIMS.dt_dx * flux);
}

  
void stepUpdate(states *state, int idx, int tstep)
{
    if (tstep & 1) //Odd - Rslt is 0 for even numbers
    {
        pressureRatio(state, idx, DIVMOD(tstep));
    }
    else
    {
        eulerStep(state, idx, DIVMOD(tstep));
    }
}


struct globalism {
    // Topology
        int nGpu, nX;
        int xg, xcpu;
        bool hasGpu;
        double gpuA;
    
    // Geometry
        int tpb, tpbp, base;
        int cBks, gBks;
        int ht, htm, htp;
        int szState;
    
    // Iterator
        double tf, freq, dt, dx, lx;
        // Initialize passing both sides.
};

globalism cGlob;

jsons inJ;
jsons solution;
jsons timing;

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

// gpuA = gBks/cBks

void initArgs()
{
    using namespace std;
    cGlob.lx = inJ["lx"].asDouble();
    cGlob.szState = sizeof(states);
    cGlob.tpb = inJ["tpb"].asInt();
    
    cGlob.dt = inJ["dt"].asDouble();
    cGlob.tf = inJ["tf"].asDouble();
    cGlob.freq = inJ["freq"].asDouble();
    cGlob.gpuA = inJ["gpuA"].asDouble();
    cGlob.nX = inJ["nX"].asInt();
    cGlob.xcpu = cGlob.nX;
    if (!cGlob.freq) cGlob.freq = cGlob.tf*2.0;


    cGlob.base = cGlob.tpb+2;
    cGlob.tpbp = cGlob.tpb+1;
    cGlob.ht = cGlob.tpb/2;
    cGlob.htm = cGlob.ht-1;
    cGlob.htp = cGlob.ht+1;
    // Derived quantities    
    inJ["xCpu"] = cGlob.xcpu;

    // Different schemes!
    cGlob.dx = cGlob.lx/(double)cGlob.nX; // Spatial step
    inJ["dx"] = cGlob.dx; // To send back to equation folder.  It may need it, it may not.

    equationSpecificArgs(inJ);
}

void solutionOutput(states *outState, double tstamp, int idx, int strt)
{
    std::string tsts = std::to_string(tstamp);
    double xpt = indexer(cGlob.dx, idx, strt);
    std::string xpts = std::to_string(xpt);
    for (int k=0; k<NVARS; k++)
    {
        solution[outVars[k]][tsts][xpts] = printout(outState + idx, k);
    }
}

/**
----------------------
    MAIN PART
----------------------
*/

#ifndef HDW
    #define HDW     "hardware/WORKSTATION.json"
#endif

std::vector<int> jsonP(jsons jp, size_t sz)
{
	std::vector <int> outv;
	for(int i=0; i<sz; i++)
	{
		outv.push_back(jp[i].asInt());
	}
	return outv;
}

void classicStepCPU(states *state, int numx, int tstep)
{
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}

double classicCPP(states *state, int xcp, int *tstep)
{
    int tmine = *tstep;
    double t_eq = 0.0;

    while (t_eq < cGlob.tf)
    {
        classicStepCPU(state, xcp, tmine);
        if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
        tmine++;
	    if (tmine % 20000 == 0) std::cout <<  tmine << " " << t_eq << " | " << cGlob.tf << std::endl;
    }
    *tstep = tmine;
    return t_eq;
}

int main(int argc, char *argv[])
{
    std::string ext = ".json";
    std::string sout = argv[3];
    sout.append(ext); 
    std::string scheme = argv[1];

    cGlob.nGpu = 0;
    // Equation, grid, affinity data
    std::ifstream injson(argv[2], std::ifstream::in);
    injson >> inJ;
    injson.close();

    parseArgs(argc, argv);
    initArgs();
    std::string pth = argv[3];

    /*
        Essentially it should associate some unique (UUID?) for the GPU with the CPU. 
        Pretend you now have a (rank, gpu) map in all memory. because you could just retrieve it with a function.
    */
    int nAlloc = cGlob.nX+2;
    states *state = new states[nAlloc];

    for (int k=0; k<nAlloc; k++)  initialState(inJ, state, k, 0);
    for (int k=1; k<nAlloc-1; k++)  solutionOutput(state, 0.0, k, 0);

    // If you have selected scheme I, it will only initialize and output the initial values.
    int tstep = 1;
    double tfm;
    clock_t t0, t1;
    t0 = clock();

    tfm = classicCPP(state, cGlob.nX + 1, &tstep);

    t1 = clock();

    double tf = (double)(t1-t0)/(CLOCKS_PER_SEC*1.0e-6);
    int tst = tstep/NSTEPS;

    std::cout << "Sub - timesteps: " << tstep << " | timestep time: " << tst*cGlob.dt << std::endl; 
    std::cout << "That took: " << tf/(double)tst << " (us) per timestep" << std::endl; 

    for (int k=1; k<nAlloc-1; k++)  solutionOutput(state, tfm, k, 0);

    std::string spath = pth + "/s" + fspec + ext;
    std::cout << spath << std:: endl;
    std::ofstream soljson(spath.c_str(), std::ofstream::trunc);
    solution["meta"] = inJ;
    soljson << solution;
    soljson.close();
    std::cout << state[0].Q[0].x << " " << state[0].Q[0].y << " " << state[0].Q[0].z << std::endl;

    delete[] state;
    return 0;
}
