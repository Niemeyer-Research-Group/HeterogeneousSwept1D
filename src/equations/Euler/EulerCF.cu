/*
    The equation specific functions.
*/

#include "EulerCF.h"

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.
    
    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
#ifdef __CUDA_ARCH__
    #define DIMS    deqConsts
#else
    #define DIMS    heqConsts
#endif

__host__ REAL density(REALthree subj)
{
    return subj.x;
}

__host__ REAL velocity(REALthree subj)
{
    return subj.y/subj.x;
}

__host__ REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

__device__ __host__ 
__forceinline__
REAL pressure(REALthree qH)
{
    return DIMS.mgam * (qH.z - (HALF * qH.y * qH.y/qH.x));
}

__host__ REAL printout(const int i, states *state)
{
    REALthree subj = state->Q[0];
    switch(i)
    {
        case 0: return density(subj);
        case 1: return velocity(subj):
        case 2: return energy(subj);
        case 3: return pressure(subj);
    } 
}

/*
dimensions heqConsts; //---------------// 
REALthree hBound[2]; // Boundary Conditions
double lx; // Length of domain.
*/

_host__ void equationSpecificArgs(json inJ)
{
    heqConsts.gammma = inJ["gamma"];
    heqConsts.mgammma = heqConsts.gammma - 1;
    REAL rhoL = inJ["rhoL"];
    REAL vL = inJ["vL"];
    REAL pL = inJ["pL"];
    REAL rhoR = inJ["rhoR"];
    REAL vR = inJ["vR"];
    REAL pR = inJ["pR"];
    hBounds[0] = {rhoL, vL*rhoL, pL/heqConsts.mgamma + HALF * rhoL * vL * vL};
    hBounds[1] = {rhoR, vR*rhoR, pR/heqConsts.mgamma + HALF * rhoR * vR * vR};
    REAL dtx = inJ["dt"];
    REAL dxx = inJ["dx"];
    heqConsts.dt_dx = dtx/dxx;
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
__host__ void initialState(json inJ, int idx, int xst, states *inl, double *xs)
{
    REAL dtx = inJ["dt"];
    REAL dxx = inJ["dx"];
    double xss = dx*(double)(idx + xst);
    xs[idx] = xss;
    bool wh = inJ["IC"] == "PARTITION";
    if (wh)
    {
        int side = (xs < HALF*lx);
        intl = hBound[side];
    }
}

__host__ void mpi_type(MPI_Datatype *dtype)
{ 
    //double 3 type
    MPI_Datatype vtype;
    MPI_Datatype typs[3] = {MPI_R, MPI_R, MPI_R};
    int n[3] = {1};
    MPI_Aint disp[3] = {0, sizeof(REAL), 2*sizeof(REAL)};

    MPI_Type_struct(3, n, disp, typs, &vtype);
    MPI_Type_commit(&vtype);

    typs[0] = vtype;
    typs[2] = vtype;
    disp[1] = 3*sizeof(vtype);
    disp[2] = 4*sizeof(REAL);

    MPI_Type_struct(3, n, disp, typs, dtype);
    MPI_Type_commit(dtype);

    MPI_Type_free(&vtype);
}

__device__ __host__ 
__forceinline__
REAL pressureRoe(REALthree qH)
{
    return DIMS.mgam * (qH.z - HALF * qH.y * qH.y);
}

/**
    Ratio
*/
__device__ __host__ 
__forceinline__
void pressureRatio(states *state, int idx, int tstep)
{
    state[idx].Pr = (pressure(state[idx+1]->Q[tstep]) - pressure(state[idx]->Q[tstep]))/(pressure(state[idx]->Q[tstep]) - pressure(state[idx-1]->Q[tstep]));
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
REALthree limitor(REALthree qH, REALthree qN, REAL pRatio)
{   
    return (isnan(pRatio) || pRatio<0) ? qH :  (qH + HALF * min(pRatio, ONE) * (qN - qH));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__ 
__forceinline__ REALthree eulerFlux(REALthree qL, REALthree qR)
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
__device__ __host__ 
__forceinline__ REALthree eulerSpectral(REALthree qL, REALthree qR)
{
    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(qL.x);
    REAL rhoRightsqrt = SQUAREROOT(qR.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*qR.y + rhoRightsqrt*qL.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*qR.z + rhoRightsqrt*qL.z)*halfDenom;

    REAL pH = pressureRoe(halfState);

    return (SQUAREROOT(pH * DIMS.gam) + fabs(halfState.y)) * (qL - qR);
}

/**
    The Final step of the finite volume scheme.

    First: The pressure ratio calculation is decomposed to avoid division and calling the limitor unnecessarily.  Although 3 pressure ratios would be required, we can see that there are only 4 unique numerators and denominators in that calculation which can be calculated without using division or calling pressure (which uses division).  The edge values aren't guaranteed to have the correct conditions so the flags set the appropriate pressure values to 0 (Pressures are equal) at the edges.
    Second:  The numerator and denominator are tested to see if the pressure ratio will be Nan or <=0. If they are, the limitor doesn't need to be called.  If they are not, call the limitor and calculate the pressure ratio.
    Third:  Use the reconstructed values at the interfaces to get the flux at the interfaces using the spectral radius and flux functions and combine the results with the flux variable.
    Fourth: Repeat for second interface and update current volume. 

    @param state  Reference to the working array in SHARED memory holding the dependent variables.
    @param idx  The indices of the stencil points.
    @param flagLeft  True if the point is the first finite volume in the tube.
    @param flagRight  True if the point is the last finite volume in the tube.
    @return  The updated value at the current spatial point.
*/
__device__ __host__ void eulerStep(states *state, int idx, int tstep)
{
    REALthree tempStateLeft, tempStateRight;

    tempStateLeft = limitor(state[idx-1].Q[tstep], state[idx].Q[tstep], state[idx-1].Pr);
    tempStateRight = limitor(state[idx].Q[tstep], state[idx-1].Q[tstep], ONE/state[idx].Pr);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Q[tstep], state[idx+1].Q[tstep], state[idx].Pr);
    tempStateRight = limitor(state[idx+1].Q[tstep], state[idx].Q[tstep], ONE/state[idx+1].Pr);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    state[idx].Q[tstep] = state[idx].Q[0] + ((QUARTER * (tstep+1)) * DIMS.dt_dx * flux);
}

__device__ __host__ 
__forceinline__ void stepUpdate(states *state, int idx, int tstep)
{
    if (tstep & 1) //Odd 0 for even numbers
    {
        pressureRatio(state, idx, DIVMOD(tstep));
    }
    else
    {
        eulerStep(state, idx, DIVMOD(tstep));
    }
}
