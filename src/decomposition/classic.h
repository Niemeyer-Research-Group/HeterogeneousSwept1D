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
    // bool ornk = omp_get_thread_num() == 0;
    // if (!ranks[1] && ornk) std::cout << "we're taking a classic step on the cpu: " << tstep << std::endl;
    for (int k=1; k<numx; k++)
    {
        stepUpdate(state, k, tstep);
    }
}

void classicPassLeft(states *state, int idxend, int tstep)
{   
    if (cGlob.bCond[0])
    {
        MPI_Isend(&state[1], 1, struct_type, ranks[0], TAGS(tstep),
                MPI_COMM_WORLD, &req[0]);
        
        if (!ranks[1]) std::cout << "we're passing a classic step Left on the cpu: "
                << tstep << " " << ranks[1] << std::endl;

        MPI_Recv(&state[0], 1, struct_type, ranks[0], TAGS(tstep+100), 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
}

void classicPassRight(states *state, int idxend, int tstep)
{
    if (cGlob.bCond[1]) 
    {
        MPI_Isend(&state[idxend-1], 1, struct_type, ranks[2], TAGS(tstep+100),
                MPI_COMM_WORLD, &req[1]);

        if (!ranks[1]) std::cout << "we're passing a classic step Right on the cpu: "
                << tstep << " " << ranks[1] << std::endl;

        MPI_Recv(&state[idxend], 1, struct_type, ranks[2], TAGS(tstep), 
                MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    }
}

// We are working with the assumption that the parallelism is too fine to see any benefit.
// Still struggling with the idea of the local vs parameter arrays.
// Classic Discretization wrapper.
double classicWrapper(states *state, int xpt, int *tstep)
{
    if (!ranks[1]) std::cout << "Classic Decomposition" << std::endl;
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        const int xc = cGlob.xcpu/2, xcp = xc+1;
        const int xcr = xc + cGlob.xg;
        const int xwrt = cGlob.xg + cGlob.xcpu;

        int nomar;
        printf("Before the first function calls\n");

        while (t_eq < cGlob.tf)
        {
            // COMPUTE
            // std::cout << state[0][5].Q[0].x << std::endl;
            classicStep<<<cGlob.bks, cGlob.tpb>>> (state + xc, tmine);

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicStepCPU(state, xcp, tmine);
                }
                #pragma omp section
                {
                    classicStepCPU(state + xcr, xcp, tmine);
                }
            }
            printf("Past the first function calls\n");
            if (tmine<8) std::cout << tmine << std::endl;
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error tstep: %i: msg %s\n", tmine, cudaGetErrorString(error));
                std::cin >> nomar;
            }

            // Host to device first. PASS
            # pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicPassRight(state + xcr, xcp, tmine);
                }
                #pragma omp section
                {
                    classicPassLeft(state, xcp, tmine);
                }
            }

            // Increment Counter and timestep
            if (MODULA(tmine)) t_eq += cGlob.dt;
            tmine++;

            // OUTPUT
            if (t_eq > twrite)
            {
                for (int k=1; k<xcp; k++) solutionOutput(state, t_eq, k, xpt);
                twrite += cGlob.freq;
            }
        }
    }
    else
    {
        const int xcp = cGlob.xcpu + 1;
        while (t_eq < cGlob.tf)
        {
            classicStepCPU(state, xcp, tmine);
            if (MODULA(tmine)) t_eq += cGlob.dt;
            tmine++;

            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    classicPassRight(state, xcp, tmine);
                }
                #pragma omp section
                {
                    classicPassLeft(state, xcp, tmine);
                }
            }

            if (t_eq > twrite)
            
                for (int k=1; k<xcp; k++) solutionOutput(state, t_eq, k, xpt); 
                twrite += cGlob.freq;
            }
    }
    *tstep = tmine;
    return t_eq;
}
