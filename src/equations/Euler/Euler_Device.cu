/*
    The equation specific functions.
*/

#include "EulerGlobals.h"

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.
    
    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
__device__ __host__ REAL pressure(REALthree qH)
{
    return dimens.mgam * (qH.z - (HALF * qH.y * qH.y/qH.x));
}

__host__ __device__ REAL pressureRoe(REALthree qH)
{
    return dimens.mgam * (qH.z - HALF * qH.y * current.y);;
}

/**
    Ratio
*/
__host__ __device__ void pressureRatio(states *state, int idx);
{
    state[idx].Pr = (pressure(state[idx+1].Q) - pressure(state[idx].Q))...
            /(pressure(state[idx].Q) - pressure(state[idx-1].Q));
}   

/**
    Ratio
*/
// __host__ __device__ void pressureRatio2(states *state, int idx);
// {
//     state[idx].Prmid = (pressure(state[idx+1].Qmid) - pressure(state[idx].Qmid))...
//                 /(pressure(state[idx].Qmid) - pressure(state[idx-1].Qmid));
// }

/**
    Reconstructs the state variables if the pressure ratio is finite and positive.

    @param cvCurrent  The state variables at the point in question.
    @param cvOther  The neighboring spatial point state variables.
    @param pRatio  The pressure ratio Pr-Pc/(Pc-Pl).
    @return The reconstructed value at the current side of the interface.
*/
__host__ __device__ REALthree limitor(REALthree qH, REALthree qN, REAL pRatio);
{   
    return (isnan(pR) || pr<0) ? qH :  (qH + HALF * min(pR,ONE) * (qN - qH));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__host__ __device__ REALthree eulerFlux(REALthree qL, REALthree qR);
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
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
__host__ __device__ REALthree eulerSpectral(REALthree qL, REALthree qR);
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif

    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(qL.x);
    REAL rhoRightsqrt = SQUAREROOT(qR.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*qR.y + rhoRightsqrt*qL.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*qR.z + rhoRightsqrt*qL.z)*halfDenom;

    REAL pH = pressureRoe(halfState);

    return (SQUAREROOT(pH*dimens.gam) + fabs(halfState.y)) * (qL - qR);
}

/**
    The predictor step of the finite volume scheme.

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
// __device__ __host__ void eulerHalfStep(REALthree *state, int idx)
// {
//     REALthree tempStateLeft, tempStateRight;

//     tempStateLeft = limitor(state[idx-1].Q, state[idx].Q, state[idx-1].Pr);
//     tempStateRight = limitor(state[idx].Q, state[idx-1].Q, ONE/state[idx].Pr);
//     REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
//     flux += eulerSpectral(tempStateLeft,tempStateRight);

//     tempStateLeft = limitor(state[idx].Q, state[idx+1].Q, state[idx].Pr);
//     tempStateRight = limitor(state[idx+1].Q, state[idx].Q, ONE/state[idx+1].Pr);
//     flux -= eulerFlux(tempStateLeft,tempStateRight);
//     flux -= eulerSpectral(tempStateLeft,tempStateRight);

//     return state[idx].Q + (QUARTER * dimens.dt_dx * flux);
// }

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep while using the predictor variables to find the flux.
__device__ __host__ void eulerFullStep(REALthree *state, int idx, int tstep)
{
    REALthree tempStateLeft, tempStateRight;

    tempStateLeft = limitor(state[idx-1].Qmid, state[idx].Qmid, state[idx-1].Prmid);
    tempStateRight = limitor(state[idx].Qmid, state[idx-1].Qmid, ONE/state[idx].Prmid);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Qmid, state[idx+1].Qmid, state[idx].Prmid);
    tempStateRight = limitor(state[idx+1].Qmid, state[idx].Qmid, ONE/state[idx+1].Prmid);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    return state[idx].Q + (HALF * dimens.dt_dx * flux);
}

__host__ __device__ void stepUpdate(states *state, int idx, int tstep)
{
    if (tstep & 1) //Even
    {

    }
    else
    {
        eulerStep();
    }

}

__host__ REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}


