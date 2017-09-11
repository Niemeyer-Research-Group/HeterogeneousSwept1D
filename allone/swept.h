
/*
---------------------------
    SWEPT CORE
---------------------------
*/

/**
    This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
*/

/**
    Builds an upright triangle using the swept rule.

    Upright triangle using the swept rule.  This function is called first using the initial conditions or after results are read out using downTriange.  In the latter case, it takes the result of down triangle as IC.

    @param initial Initial condition.
    @param tstep The timestep we're starting with.
*/

static int offSend[2];
static int offRecv[2];
static int cnt, turn;

void swIncrement()
{
    cnt++;
    turn = cnt & 1;
}

__global__ void upTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;

    // Using tidx as tid is kind of confusing for reader but looks valid.

	temper[tidx] = state[gid + 1];

    __syncthreads();

	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tstep + k); 
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx];
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
*/
__global__
void
downTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
    __syncthreads();
	temper[tid+2] = state[gid+2];
    
    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid] = temper[tidx];
}


/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param state The working array of structures states.
    @param tstep The count of the first timestep.
*/
__global__
void
wholeDiamond(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
    __syncthreads();
	temper[tid + 2] = state[gid + 2];

    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tstep - k);
		}
		__syncthreads();
	}

	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tstep + (k-2));
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx];
}

// HOST SWEPT ROUTINES
void upTriangleCPU(states *state, int tstep)
{
    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}


void downTriangleCPU(states *state, int tstep, int zi, int zf)
{
    for (int k=cGlob.ht; k>1; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
    for (int n=zi; n<zf; n++)
    {
        stepUpdate(state, n, tstep);
    }
}

// Now you can call this on a split for all proc/threads except (0,0)
void wholeDiamondCPU(states *state, int tstep, int zi, int zf)
{
    for (int k=cGlob.ht; k>1; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }

    for (int n=zi; n<zf; n++)
    {
        stepUpdate(state, n, tstep);
    } 

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tstep + (k-1));
        }
    }
}

void splitDiamondCPU(states *state, int tstep)
{
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tstep + (k-1));
            }
        }
    }
}

static void inline passSwept(states *stateSend, states *stateRecv, int tstep)
{
    MPI_Isend(stateSend + offSend[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD, &req[turn]);

    MPI_Recv(stateRecv + offRecv[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep), 
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
}

/*
    The idea of the timestep counter is: the timestep you're on is the timestep you would export 
    if you called downTriangle NEXT.

*/

double sweptWrapper(states **state, double **xpts, int *tstep)
{
    if (!ranks[1]) std::cout << "Swept Decomposition" << std::endl;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    int tmine = *tstep;

    states **staten = new states* [cGlob.nThreads]; 
    int ar1, ptin, tid, strt; 
    const int lastThread = cGlob.nThreads-1, lastWave = cGlob.nWaves-1;
    int sw[2] = {!ranks[1], ranks[1] == lastproc}; // {First node, last node
    int stride;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        int tps = cGlob.nThreads/2; // Threads per side.
        for (int k=0; k<cGlob.nThreads; k++)
        {
            ar1 = (k/tps) * 2;
            ptin = (k % tps) * cGlob.tpb; 
            staten[k] = (state[ar1] + ptin); // ptr + offset
        }

        stride = tps * cGlob.tpb;
        const int xc =cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;

        const size_t gpusize = cGlob.szState * (xgpp + cGlob.htp);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;

        offSend[0] = 1, offSend[1] = xc; // {left, right}    
        offRecv[0] = xcp, offRecv[1] = 0; 

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        std::cout << "GPU threads launched: " << cGlob.tpb*cGlob.bks << " GpuSize: " << gpusize/cGlob.szState << "  " << ptsize/cGlob.szState << " node ht: " << cGlob.ht << std::endl;

        states *dState; 

        cudaMalloc((void **)&dState, gpusize);

        cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice);
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error MEMCPY: %s\n", cudaGetErrorString(error));
        }
        /* -- DOWN MUST FOLLOW A SPLIT AND UP CANNOT BE IN WHILE LOOP SO DO UP AND FIRST SPLIT OUTSIDE OF LOOP 
            THEN LOOP CAN BE WITH WHOLE - DIAMOND - CHECK DOWN.
        */
        // ------------ Step Forward ------------ //
        // ------------ UP ------------ //

        upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, tmine);
        states *dState2 = dState + cGlob.ht;
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error UPTRIANGLE: %s\n", cudaGetErrorString(error));
        }

        // The pointers are already strided, now we stride each wave of pointers.
        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
        {
            tid = omp_get_thread_num();
            for (int k=0; k<cGlob.nWaves; k++)
            {
                strt = k*stride;
                upTriangleCPU(staten[tid] + strt, tmine);
            }
        }

        // ------------ Pass Edges ------------ // Give to 0, take from 2.
        // I'm doing it backward but it's the only way that really makes sense in this context.
        // Before I really abstracted this process away.  But here I have to deal with it.  
        //If you pass left first, you need to start at ht rather than 0.  That's the problem.

        passSwept(state[0], state[2], tmine);
        cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
        cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
        swIncrement();//Increment

        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error UPPASS: %s\n", cudaGetErrorString(error));
        }

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState2, tmine);

        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
        {
            tid = omp_get_thread_num();
            sw[1] = (sw[1] && tid == lastThread); // Only true for last proc and last thread.
            for (int k=0; k<cGlob.nWaves; k++)
            {
                sw[1] = (sw[1] && k == lastWave);
                strt = k*stride + cGlob.ht;
                if (sw[1])
                {
                    splitDiamondCPU(staten[0] + strt, tmine);
                }
                else
                {
                    wholeDiamondCPU(staten[tid] + strt, tmine, 1, cGlob.tpbp);
                }
            }
        }   
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error SPLIT: %s\n", cudaGetErrorString(error));
        }

        // ------------ Pass Edges ------------ //
        
        passSwept(state[2], state[0], tmine);
        cudaMemcpyAsync(state[2] + offRecv[turn], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost, st1);
        cudaMemcpyAsync(dState, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
        swIncrement();

        // Increment Counter and timestep
        tmine += cGlob.tpb;
        t_eq += cGlob.dt * (tmine/NSTEPS);

        if (!ranks[1]) 
        {
            std::cout << "Just for fun: " << xcpp << " nums in cpu " << xgpp << " nums in GPU " << cGlob.hasGpu << std::endl; 
            std::cout << "AND the ends of x: " << xpts[0][xc] << " " << xpts[1][xc] << " " << xpts[2][xc] << std::endl; 
        }

        while(t_eq < cGlob.tf)
        {
            // ------------ Step Forward ------------ //
            // ------------ WHOLE ------------ //
            
            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState, tmine);

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
            {
                tid = omp_get_thread_num();
                sw[0] = (sw[0] && !tid);
                sw[1] = (sw[1] && tid == lastThread);
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    sw[0] = (sw[0] && !k);
                    sw[1] = (sw[1] && tid == lastWave);   
                    strt = k*stride;
                    wholeDiamondCPU(staten[tid] + strt, tmine, sw[0] + 1,  cGlob.tpbp-sw[1]);
                }
            }   
            error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error WHOLE: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            // ------------ Pass Edges ------------ //
            
            passSwept(state[0], state[2], tmine);
            cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
            swIncrement();//Increment

            // Increment Counter and timestep
            tmine += cGlob.tpb;
            t_eq += cGlob.dt * (tmine/NSTEPS);

            // ------------ Step Forward ------------ //
            // ------------ SPLIT ------------ //

            wholeDiamond <<<cGlob.bks, cGlob.tpb, smem>>> (dState + cGlob.ht, tmine);

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
            {
                tid = omp_get_thread_num();
                sw[1] = (sw[1] && tid == lastThread); // Only true for last proc and last thread.
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    sw[1] = (sw[1] && k == lastWave);
                    strt = k*stride + cGlob.ht;
                    if (sw[1])
                    {
                        splitDiamondCPU(staten[0] + strt, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(staten[tid] + strt, tmine, 1, cGlob.tpbp);
                    }
                }
            }   

            // ------------ Pass Edges ------------ //
            
            passSwept(state[2], state[0], tmine);
            cudaMemcpyAsync(state[2] + offRecv[turn], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost, st1);
            cudaMemcpyAsync(dState, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
            swIncrement();

            // Increment Counter and timestep
            tstep += cGlob.tpb;
            t_eq += cGlob.dt * (tmine/NSTEPS);

            if (t_eq > twrite)
            {
                downTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState + cGlob.htp, tmine);

                #pragma parallel num_threads(cGlob.nThreads) private(strt, tid, sw)
                {
                    tid = omp_get_thread_num();
                    sw[0] = (sw[0] && !tid);
                    sw[1] = (sw[1] && tid == lastThread);
                    for (int k=0; k<cGlob.nWaves; k++)
                    {
                        sw[0] = (sw[0] && !k);
                        sw[1] = (sw[1] && tid == lastWave);   
                        strt = k*stride;
                        downTriangleCPU(staten[tid] + strt, tmine, sw[0] + 1,  cGlob.tpbp-sw[1]);
                    }
                }   

                cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);
                                
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);                
                for (int k=1; k<xcp; k++) solutionOutput(state[2]+k, xpts[2][k], t_eq);
                for (int k=1; k<xgp; k++) solutionOutput(state[1]+k, xpts[1][k], t_eq);

                upTriangle <<<cGlob.bks, cGlob.tpb, smem>>> (dState, tmine);

                // The pointers are already strided, now we stride each wave of pointers.
                #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
                {
                    tid = omp_get_thread_num();
                    for (int k=0; k<cGlob.nWaves; k++)
                    {
                        strt = k*stride;
                        upTriangleCPU(staten[tid] + strt, tmine);
                    }
                }

                passSwept(state[0], state[2], tmine);
                cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
                cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
                swIncrement();//Increment

                // Increment Counter and timestep
                tstep += cGlob.tpb;
                t_eq += cGlob.dt * (tmine/NSTEPS);

                twrite += cGlob.freq;
            }
        }
        cudaFree(dState);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
    }
    // YES YOU CAN PASS THE SAME POINTER AS TWO DIFFERENT ARGS TO THE FUNCITON.
    else
    {
        int xcp = cGlob.xcpu + cGlob.htp;
        
        for (int k=0; k<cGlob.nThreads; k++)
        {
            ptin = k * cGlob.tpb; 
            staten[k] = (state[0] + ptin); // ptr + offset
        }

        #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
        {
            tid = omp_get_thread_num();
            for (int k=0; k<cGlob.nWaves; k++)
            {
                strt = k*stride;
                upTriangleCPU(staten[tid] + strt, tmine);
            }
        }

        passSwept(state[0], state[0], tmine); // Left first
        swIncrement();

        while (t_eq < cGlob.tf)
        {

            #pragma parallel num_threads(cGlob.nThreads) private(strt, tid)
            {
                tid = omp_get_thread_num();
                for (int k=0; k<cGlob.nWaves; k++)
                {
                    strt = k*stride;
                    if (!ranks[1] && !(tid + k))
                    {
                        splitDiamondCPU(staten[0], tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(staten[tid] + strt, tmine, 1, cGlob.tpbp);
                    }
                }
            }   

            passSwept(state[0], state[0], tmine); // Left first
            swIncrement();

            if (t_eq > twrite)
            {
                #pragma omp parallel for
                for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
                twrite += cGlob.freq;
            }
            
        }
        #pragma omp parallel for
        for (int k=1; k<xcp; k++) solutionOutput(state[0]+k, xpts[0][k], t_eq);
    }
    *tstep = tmine;
    return t_eq;
}