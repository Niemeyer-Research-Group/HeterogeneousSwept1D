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
}

// Blocks because one is called and then the other so the PASS blocks.
void classicPass(states *putst, states *getst, int tstep)
{   
    int t0 = TAGS(tstep), t1 = TAGS(tstep + 100);

    MPI_Isend(putst[0], 1, struct_type, ranks[0], t0, MPI_COMM_WORLD, &req[0]);

    MPI_Isend(putst[1], 1, struct_type, ranks[2], t1, MPI_COMM_WORLD, &req[1]);

    MPI_Irecv(getst[1], 1, struct_type, ranks[2], t0, MPI_COMM_WORLD, &req[0]);

    MPI_Irecv(getst[0], 1, struct_type, ranks[0], t1, MPI_COMM_WORLD, &req[1]); 

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request_free(&req[0]);
    MPI_Request_free(&req[1]);    
}

/*   
if (tstep < 5) std::cout << "Start: " << ranks[1] << " " << tstep << " " << stateL[0].T[0] << " " << stateR[idxend].T[0] << std::endl; 

if (tstep < 5) std::cout << "End: " << ranks[1] << " " << tstep << " " << stateL[0].T[0]  << " " << stateR[idxend].T[0]  << std::endl;
/*
MPI_Isend(stateL + 1, 1, struct_type, ranks[0], t0, MPI_COMM_WORLD, &req[0]);
MPI_Isend(stateR + (idxend-1), 1, struct_type, ranks[2], t1, MPI_COMM_WORLD, &req[1]);
MPI_Irecv(stateR + idxend, 1, struct_type, ranks[2], t0, MPI_COMM_WORLD, &req[0]);
MPI_Irecv(stateL, 1, struct_type, ranks[0], t1, MPI_COMM_WORLD, &req[1]); 
*/

// We are working with the assumption that the parallelism is too fine to see any benefit.
// Classic Discretization wrapper.
double classicWrapper(states **state, ivec xpts, ivec alen, int *tstep)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    states putst[2], getst[2];

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        std::cout << "Classic Decomposition GPU" << std::endl;
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

        while (t_eq < cGlob.tf)
        {
            classicStep<<<cGlob.gBks, cGlob.tpb>>> (dState, tmine);
            classicStepCPU(state[0], xcp, tmine);
            classicStepCPU(state[2], xcp, tmine);

            cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
            cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
            cudaMemcpyAsync(state[2], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4); 
            cudaDeviceSynchronize();
            putst[0] = state[0][1], putst[1] = state[1][xc]; 

            classicPass(&puts, &gets, tmine);
            if (cGlob.bCond[0]) state[0][0] = gets[0]; 
            if (cGlob.bCond[1]) state[2][xcp] = gets[1];

            // Increment Counter and timestep
            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

                for (int i=0; i<3; i++)
                {
                    for (int k=1; k<alen[i]; k++)  solutionOutput(state[i], t_eq, k, xpts[i]);
                }  

                twrite += cGlob.freq;
            }
        }       

        if (!(tmine % 20000)) std::cout << tmine << " | " << t_eq << " | " << std::endl;
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

            putst[0] = state[0][1], putst[1] = state[1][xc]; 
            classicPass(&puts, &gets, tmine);
            if (cGlob.bCond[0]) state[0][0] = gets[0]; 
            if (cGlob.bCond[1]) state[2][xcp] = gets[1];

            if (!(tmine % NSTEPS)) t_eq += cGlob.dt;
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