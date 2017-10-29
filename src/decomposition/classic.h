/**
---------------------------
    CLASSIC CORE
---------------------------
*/

/**
    The Classic Functions for the stencil operation
*/

/** 
    Classic kernel for simple decomposition of spatial domain.

    @param States The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
using namespace std;

typedef std::vector<int> ivec;

__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)
    stepUpdate(state, gid, ts);
}

void classicStepCPU(states *state, int numx, int tstep)
{
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
    if ((tstep>120 && tstep <125) || tstep<6 ) cout << ranks[1] << " " << printout(state, 0) << " " << printout(state+1, 0) << " " << tstep << " " << printout(state+numx-10, 0) << " " << printout(state + numx, 0) << endl; 
}

// void classicDPass(double *putSt, double *getSt, int tstep)
// {   
//     int t0 = TAGS(tstep), t1 = TAGS(tstep + 100);
//     int rnk;

//     MPI_Isend(putSt, 1, MPI_DOUBLE, ranks[0], t0, MPI_COMM_WORLD, &req[0]);

//     MPI_Isend(putSt + 1, 1, MPI_DOUBLE, ranks[2], t1, MPI_COMM_WORLD, &req[1]);

//     MPI_Irecv(getSt + 1, 1, MPI_DOUBLE, ranks[2], t0, MPI_COMM_WORLD, &req[0]);

//     MPI_Irecv(getSt, 1, MPI_DOUBLE, ranks[0], t1, MPI_COMM_WORLD, &req[1]);

//     MPI_Barrier(MPI_COMM_WORLD);
//     // // MPI_Request_free(&req[0]);
//     // // MPI_Request_free(&req[1]); 

//     MPI_Wait(&req[0], &stat[0]);
//     MPI_Wait(&req[1], &stat[1]);
//     MPI_Get_count(&stat[0], MPI_DOUBLE, &t0);
//     MPI_Get_count(&stat[1], MPI_DOUBLE, &t1);
//     if (tstep < 10) 
//     {
//         cout << "Exit Pass: " << tstep << " rnk: " << ranks[1] << "  p: " << putSt[0] << " " << putSt[1] << "\ng: " << getSt[0] << " " << getSt[1] << endl;
//         cout << "Exit Status: " << tstep << " rnk: " << ranks[1] << "  0: " << t0 << " 1: " << t1 << endl;
//     }    
// }

// Blocks because one is called and then the other so the PASS blocks.
void passClassic(REAL *puti, REAL *geti, int tstep)
{   
    int t0 = TAGS(tstep), t1 = TAGS(tstep + 100);
    int rnk;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Isend(puti, NSTATES, MPI_R, ranks[0], t0, MPI_COMM_WORLD, &req[0]);

    MPI_Isend(puti + NSTATES, NSTATES, MPI_R, ranks[2], t1, MPI_COMM_WORLD, &req[1]);

    MPI_Recv(geti + NSTATES, NSTATES, MPI_R, ranks[2], t0, MPI_COMM_WORLD, &stat[0]);

    MPI_Recv(geti, NSTATES, MPI_R, ranks[0], t1, MPI_COMM_WORLD, &stat[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Wait(&req[0], &stat[0]);
    MPI_Wait(&req[1], &stat[1]);
}


// Classic Discretization wrapper.
double classicWrapper(states **state, ivec xpts, ivec alen, int *tstep)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    // Must be declared global in equation specific header.
    stPass = 2; 
    numPass = NSTATES * stPass;

    states putSt[stPass];
    states getSt[stPass];
    REAL putRe[numPass];
    REAL getRe[numPass];
 
    int t0, t1;
    int nomar;
    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        cout << "Classic Decomposition GPU" << endl;
        const int xc = cGlob.xcpu/2;
        int xcp = xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;

        states *dState;
        
        cudaMalloc((void **)&dState, gpusize);
        // Copy the initial conditions to the device array.
        // This is ok, the whole array has been malloced.
        cudaMemcpy(dState, state[1], gpusize, cudaMemcpyHostToDevice);

        // Four streams for four transfers to and from cpu.
        cudaStream_t st1, st2, st3, st4;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);
        cudaStreamCreate(&st4);      

        cout << "Entering Loop" << endl;

        while (t_eq < cGlob.tf)
        {
            classicStep<<<cGlob.gBks, cGlob.tpb>>> (dState, tmine);
            classicStepCPU(state[0], xcp, tmine);
            classicStepCPU(state[2], xcp, tmine);

            cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
            cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
            cudaMemcpyAsync(state[2], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4); 

            putSt[0] = state[0][1];
            putSt[1] = state[2][xc]; 
            unstructify(&putSt[0], &putRe[0]);
            passClassic(&putRe[0], &getRe[0], tmine);
            restructify(&getSt[0], &getRe[0]);
            if (cGlob.bCond[0]) state[0][0] = getSt[0]; 
            if (cGlob.bCond[1]) state[2][xcp] = getSt[1];

            // if (tmine > 50 && tmine < 53)
            // {
            //     for (int i=0; i<3; i++)
            //     {
            //         cout << "ARRAY " << i << endl;
            //         for (int k=0; k<=xcp; k++) cout << " " << printout(state[i] + k, 0);
            //         cout << endl;
            //     }
            // }

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

                for (int i=0; i<3; i++)
                {
                    for (int k=0; k<=alen[i]; k++)  solutionOutput(state[i], t_eq, k, xpts[i]);
                }          

                twrite += cGlob.freq;
            }
            if (!(tmine % 200000)) std::cout << tmine << " | " << t_eq << " | " << std::endl;
        }       

        cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        cudaStreamDestroy(st4);
        
        cudaFree(dState);
    }
    else
    {
        int xcp = cGlob.xcpu + 1;
        while (t_eq < cGlob.tf)
        {
            classicStepCPU(state[0], xcp, tmine);

            putSt[0] = state[0][1];
            putSt[1] = state[0][cGlob.xcpu]; 
            unstructify(&putSt[0], &putRe[0]);
            passClassic(&putRe[0], &getRe[0], tmine);
            restructify(&getSt[0], &getRe[0]);
            if (cGlob.bCond[0]) state[0][0] = getSt[0]; 
            if (cGlob.bCond[1]) state[0][xcp] = getSt[1];

            // cout << "CPU ARRAY " << ranks[1]  << " | " << tmine << "\n" << "p: " << printout(&putSt[0], 0) << " | " << printout(&putSt[1], 1) << "\ng: " << printout(&getSt[0], 0) << " | " << printout(&getSt[1], 1) << endl;

            if (!(tmine % NSTEPS)) 
            {
                t_eq += cGlob.dt;
            }
            tmine++;

            if (t_eq > twrite)
            {
                for (int k=1; k<alen[0]; k++)  solutionOutput(state[0], t_eq, k, xpts[0]);
                twrite += cGlob.freq;
            }
        }   
    }
    *tstep = tmine;
    return t_eq;
}