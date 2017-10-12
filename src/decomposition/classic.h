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
void classicPass(states *stateL, states *stateR, int idxend, int tstep)
{   
    
    
    if (cGlob.bCond[0]) MPI_Isend(&stateL[1], 1, struct_type, ranks[0], TAGS(tstep),
            MPI_COMM_WORLD, &req[0]);

    if (cGlob.bCond[1]) MPI_Isend(&stateR[idxend-1], 1, struct_type, ranks[2], TAGS(tstep+100),
            MPI_COMM_WORLD, &req[1]);
    
    if (!ranks[1]) std::cout << "we're passing a classic step Left on the cpu: "
            << tstep << " " << ranks[1] << std::endl;
    
    if (cGlob.bCond[0]) MPI_Recv(&stateR[idxend], 1, struct_type, ranks[2], TAGS(tstep), 
                MPI_COMM_WORLD,  MPI_STATUS_IGNORE);

    if (cGlob.bCond[1]) MPI_Recv(&stateL[0], 1, struct_type, ranks[0], TAGS(tstep+100), 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

}

// We are working with the assumption that the parallelism is too fine to see any benefit.
// Still struggling with the idea of the local vs parameter arrays.
// Classic Discretization wrapper.
double classicWrapper(states **state, std::vector<int> xpts, std::vector<int> alen, int *tstep)
{
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        std::cout << "Classic Decomposition GPU" << std::endl;
        const int xc = cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int gpusize =  cGlob.szState * xgpp;
        const int cpuzise = cGlob.szState * xcpp;

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

        std::cout << "Classic - streams initialized and memcpied." << std::endl;  

        while (t_eq < cGlob.tf)
        {
            classicStep<<<cGlob.gBks, cGlob.tpb>>> (dState, tmine);
            classicStepCPU(state[0], xcp, tmine);
            classicStepCPU(state[2], xcp, tmine);
            if (tmine<3)  std::cout << "Classic - Complete GPU timestep: " << tmine << std::endl;

            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error tstep: %i: msg %s\n", tmine, cudaGetErrorString(error));
            }


            cudaMemcpyAsync(dState, state[0] + xc, cGlob.szState, cudaMemcpyHostToDevice, st1);
            cudaMemcpyAsync(dState + xgp, state[2] + 1, cGlob.szState, cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(state[0] + xcp, dState + 1, cGlob.szState, cudaMemcpyDeviceToHost, st3);
            cudaMemcpyAsync(state[2], dState + cGlob.xg, cGlob.szState, cudaMemcpyDeviceToHost, st4); 
            classicPass(state[0], state[2], xcp, tmine);
            if (tmine<3)  std::cout << "Classic - Complete GPU Pass: " << tmine << std::endl;

            // Increment Counter and timestep
            if (MODULA(tmine)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost);

                for (int i=0; i<3; i++)
                {
                    for (int k=1; k<=alen[i]; k++)  solutionOutput(state[i], t_eq, k, xpts[i]);
                }  

                twrite += cGlob.freq;
            }
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
        if (!ranks[1])  std::cout << "Classic - Init CPU only: " << tmine << std::endl;

        int xcp = cGlob.xcpu + 1;
        while (t_eq < cGlob.tf)
        {
            classicStepCPU(state[0], xcp, tmine);
            if (MODULA(tmine)) t_eq += cGlob.dt;
            tmine++;

            if (!ranks[1] && tmine<3)  std::cout << "Classic - CPU Complete Timestep: " << tmine << std::endl;

            classicPassRight(state[0], xcp, tmine);

            classicPassLeft(state[0], xcp, tmine);

            if (t_eq > twrite)
            {
                for (int k=1; k<=cGlob.xcpu; k++)  solutionOutput(state[0], t_eq, k, xpts[0]);
                twrite += cGlob.freq;
            }
        }   
    }
    *tstep = tmine;
    return t_eq;
}