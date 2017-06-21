#include "Euler_Device.cuh"


// This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
#ifndef REAL
    #define REAL            float
    #define REALtwo         float2
    #define REALthree       float3
    #define SQUAREROOT(x)   sqrtf(x)

    #define ZERO            0.0f
    #define QUARTER         0.25f
    #define HALF            0.5f
    #define ONE             1.f
    #define TWO             2.f
#else

    #define ZERO            0.0
    #define QUARTER         0.25
    #define HALF            0.5
    #define ONE             1.0
    #define TWO             2.0
    #define SQUAREROOT(x)   sqrt(x)
#endif

// Hardwire in the length of the 
const REAL lx = 1.0;

// The structure to carry the initial and boundary conditions.
// 0 is left 1 is right.
REALthree bd[2];

//dbd is the boundary condition in device constant memory.
__constant__ REALthree dbd[2]; 

//Protoype for useful information struct.
struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int base; // Length of node + stencils at end (4)
    int idxend; // Last index (number of spatial points - 1)
    int idxend_1; // Num spatial points - 2
    int hts[5]; // The five point stencil around base/2
};

// structure of dimensions in cpu memory
dimensions dimz;

// Useful and efficient to keep the important constant information in GPU constant memory.
__constant__ dimensions dimens;

/**
    Takes the passed the right and left arrays from previous cycle and inserts them into new SHARED memory working array.  
    
    Called at start of kernel/ function.  The passed arrays have already been passed readIn only finds the correct index and inserts the values and flips the indices to seed the working array for the cycle.
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param temp  The working array in shared memory
*/
__host__ __device__
__forceinline__
void
readIn(REALthree *temp, const REALthree *rights, const REALthree *lefts, int td, int gd)
{
    // The index in the SHARED memory working array to place the corresponding member of right or left.
    #ifdef __CUDA_ARCH__  // Accesses the correct structure in constant memory.
	int leftidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + ((td>>2)<<1) + (td & 3);
    #else
    int leftidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + (td & 3) - (4 + ((td>>2)<<1));
    int rightidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + ((td>>2)<<1) + (td & 3);
    #endif

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

/**
    Write out the right and left arrays at the end of a kernel when passing right.  
    
    Called at end of the kernel/ function.  The values of the working array are collected in the right and left arrays.  As they are collected, the passed edge (right) is inserted at an offset. This function is never called from the host so it doesn't need the preprocessor CUDA flags.'
    @param temp  The working array in shared memory
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param bd  The number of threads in a block (spatial points in a node).
*/
__device__
__forceinline__
void
writeOutRight(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    int gdskew = (gd + bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2; 
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}

/**
    Write out the right and left arrays at the end of a kernel when passing left.  
    
    Called at end of the kernel/ function.  The values of the working array are collected in the right and left arrays.  As they are collected, the passed edge (left) is inserted at an offset. 
    @param temp  The working array in shared memory
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param bd  The number of threads in a block (spatial points in a node).
*/
__host__ __device__
__forceinline__
void
writeOutLeft(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    #ifdef __CUDA_ARCH__
    int gdskew = (gd - bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
    #else
    int gdskew = gd;
    int leftidx = (((td>>2) & 1)  * dimz.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimz.base-6) + (((td>>2) & 1)  * dimz.base) + (td & 3) - ((td>>2)<<1);
    #endif

    rights[gd] = temp[rightidx];
    lefts[gdskew] = temp[leftidx];
}

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.
    
    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
__device__ __host__
__forceinline__
REAL
pressure(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #else
    return dimz.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #endif
}

/**
    Calculates the parameter for the first term in the spectral radius formula (P/rho).
    
    Since this formula is essentially a lambda for a single calculation, input vector y and z are u_sp and e_sp respectively without multipliation by rho and it returns the pressure over rho to skip the next step.
    @param current  The Roe averaged state variables at the interface.  Where y and z are u_sp and e_sp respectively without multipliation by rho.
    @return Roe averaged pressure over rho at the interface 
*/
__device__ __host__
__forceinline__
REAL
pressureHalf(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - HALF * current.y * current.y);
    #else
    return dimz.mgam * (current.z - HALF * current.y * current.y);
    #endif
}

/**
    Reconstructs the state variables if the pressure ratio is finite and positive.

    @param cvCurrent  The state variables at the point in question.
    @param cvOther  The neighboring spatial point state variables.
    @param pRatio  The pressure ratio Pr-Pc/(Pc-Pl).
    @return The reconstructed value at the current side of the interface.
*/
__device__ __host__
__forceinline__
REALthree
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    return (cvCurrent + HALF * min(pRatio,ONE) * (cvOther - cvCurrent));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param cvLeft Reconstructed value at the left side of the interface.
    @param cvRight  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__
__forceinline__
REALthree
eulerFlux(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;

    REAL pL = pressure(cvLeft);
    REAL pR = pressure(cvRight);

    REALthree flux;
    flux.x = (cvLeft.y + cvRight.y);
    flux.y = (cvLeft.y*uLeft + cvRight.y*uRight + pL + pR);
    flux.z = (cvLeft.z*uLeft + cvRight.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}

/**
    Finds the spectral radius and applies it to the interface.

    @param cvLeft Reconstructed value at the left side of the interface.
    @param cvRight  Reconstructed value at the left side of the interface.
    @return  The spectral radius multiplied by the difference of the reconstructed values
*/
__device__ __host__
__forceinline__
REALthree
eulerSpectral(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif

    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(cvLeft.x);
    REAL rhoRightsqrt = SQUAREROOT(cvRight.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*cvRight.y + rhoRightsqrt*cvLeft.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*cvRight.z + rhoRightsqrt*cvLeft.z)*halfDenom;

    REAL pH = pressureHalf(halfState);

    #ifdef __CUDA_ARCH__
    return (SQUAREROOT(pH*dimens.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #else
    return (SQUAREROOT(pH*dimz.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #endif
}

/**
    The predictor step of the finite volume scheme.

    First: The pressure ratio calculation is decomposed to avoid division and calling the limitor unnecessarily.  Although 3 pressure ratios would be required, we can see that there are only 4 unique numerators and denominators in that calculation which can be calculated without using division or calling pressure (which uses division).  The edge values aren't guaranteed to have the correct conditions so the flags set the appropriate pressure values to 0 (Pressures are equal) at the edges.
    Second:  The numerator and denominator are tested to see if the pressure ratio will be Nan or <=0. If they are, the limitor doesn't need to be called.  If they are not, call the limitor and calculate the pressure ratio.
    Third:  Use the reconstructed values at the interfaces to get the flux at the interfaces using the spectral radius and flux functions and combine the results with the flux variable.
    Fourth: Repeat for second interface and update current volume. 
    @param state  Reference to the working array in SHARED memory holding the dependent variables.
    @param tr  The indices of the stencil points.
    @param flagLeft  True if the point is the first finite volume in the tube.
    @param flagRight  True if the point is the last finite volume in the tube.
    @return  The updated value at the current spatial point.
*/
__device__ __host__
REALthree
eulerStutterStep(REALthree *state, int tr, char flagLeft, char flagRight)
{
    //P1-P0
    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;
    //P2-P1
    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));
    //P3-P2
    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));
    //P4-P3
    REAL pRR = (flagRight) ? ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));

    //This is the temporary state bounded by the limitor function.
    //Pr0 = PL/PLL*rho0/rho2  Pr0 is not -, 0, or nan.
    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan, pass Pr1^-1.
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    //Pressure needs to be recalculated for the new limited state variables.
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan.
    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    //Pr2 = PRR/PR*rho2/rho4  Pr2 is not - or nan, pass Pr2^-1.
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0)) ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    return state[tr] + (QUARTER * dimens.dt_dx * flux);
    #else
    return state[tr] + (QUARTER * dimz.dt_dx * flux);
    #endif

}

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep while using the predictor variables to find the flux.
__device__ __host__
REALthree
eulerFinalStep(REALthree *state, int tr, char flagLeft, char flagRight)
{

    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;

    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));

    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));

    REAL pRR = (flagRight) ?  ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));

    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0))  ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    // Return only the RHS of the discretization.
    #ifdef __CUDA_ARCH__
    return (HALF * dimens.dt_dx * flux);
    #else
    return (HALF * dimz.dt_dx * flux);
    #endif

}