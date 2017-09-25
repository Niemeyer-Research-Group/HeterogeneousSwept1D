
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

using namespace std;

int offSend[2];
int offRecv[2];
int cnt, turn;

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
downTriangle(states *state, int tstep, int offset)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
    int gid = blockDim.x * blockIdx.x + tid + offset; 
    int tidx = tid + 1;

    int tnow = tstep; // read tstep into register.

    if (tid<2) temper[tid] = state[gid]; 
    __syncthreads();
    temper[tid+2] = state[gid+2];
    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
	            stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}
    state[gid] = temper[tidx];
}

// __global__
// void
// printgpu(states *state, int n, int on)
// {
//     printf("%i\n", on);
//     for (int k=0; k<n; k++) printf("%i %.2f\n", k, state[k].T[on]);
// }

/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param state The working array of structures states.
    @param tstep The count of the first timestep.
*/
__global__
void
wholeDiamond(states *state, int tstep, int offset)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid + offset;
    int tidx = tid + 1;

    int tnow = tstep;
    if (tid<2) temper[tid] = state[gid]; 
    __syncthreads();
    temper[tid + 2] = state[gid + 2];

    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
        	stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}

    // printf("After: %i - %i - %i - %.2f \n", gid, blockIdx.x, tid, temper[tid+2].T[0]);

	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
    }
    
    state[gid + 1] = temper[tidx];
}

/*
    MARK : HOST SWEPT ROUTINES
*/

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

void downTriangleCPU(states *state, int tstep)
{
    int tnow=tstep;
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
	tnow++;
    }
}

// Now you can call this on a split for all proc/threads except (0,0)
void wholeDiamondCPU(states *state, int tstep)
{
    int tnow=tstep;
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
	tnow++;
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
    tnow++;
    }
}

void splitDiamondCPU(states *state, int tstep)
{
    int tnow=tstep;
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tnow);
            }
        }
        tnow++;
    }

    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n < cGlob.ht || n > cGlob.htp)
            {
                stepUpdate(state, n, tnow);
            }
        }
	tnow++;
    }
}

static void inline passSwept(states *stateSend, states *stateRecv, int tstep)
{
    MPI_Isend(stateSend + offSend[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD, &req[turn]);

    MPI_Recv(stateRecv + offRecv[turn], cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
}

// void applyBC(states *state, int ty, int pt)
// {
//     // Like if-dirichilet
//     // Works for whole 
//     state[ty*pt] = sBound[ty]; 
//     // If reflective
//     // state[ty*pt] = state[pt-2] or state[pt+2]
// }

/*
    The idea of the timestep counter is: the timestep you're on is the timestep you would export
    if you called downTriangle NEXT.

*/
// Now we need to put the last value in a bucket, and append that to the start of the next array.

double sweptWrapper(states **state, std::vector<int> xpts, std::vector<int> alen, int *tstep)
{
    if (!ranks[1]) std::cout << "Swept Decomposition" << std::endl;
    int tmine = *tstep;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    


    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
    
        const int xc = cGlob.xcpu/2, xcp = xc+1, xcpp = xc+2;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int cmid = cGlob.cBks/2;
        int ix;

        const size_t gpusize = cGlob.szState * (xgpp + cGlob.ht);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;

        offSend[0] = 1, offSend[1] = xc; // {left, right}
        offRecv[0] = xcp, offRecv[1] = 0;

        int gpupts = gpusize/cGlob.szState;
        int passpt = passsize/cGlob.szState;

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);

        std::cout << "GPU threads launched: " << cGlob.tpb*cGlob.gBks << " GpuSize: " << gpupts << "  " << ptsize/cGlob.szState << " node ht: " << cGlob.ht << " sizeof states: " << cGlob.szState << std::endl;
        
        cout << "AFTER Initial copy " << endl;

        states *dState;

        cudaCheckError(cudaMalloc((void **)&dState, gpusize));

        cudaCheckError(cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice));

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

        upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error UPTRIANGLE: %s\n", cudaGetErrorString(error));
        }

        for (int k=0; k<cGlob.cBks; k++)
        { 
            ix = 2*(k/cmid);
            upTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
        }

        cout << "After UPTRIANGLE " << endl;

        cudaDeviceSynchronize();

        cudaCheckError(cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost));

        // ------------ Pass Edges ------------ // Give to 0, take from 2.
        // I'm doing it backward but it's the only way that really makes sense in this context.
        // Before I really abstracted this process away.  But here I have to deal with it.
        //If you pass left first, you need to start at ht rather than 0.  That's the problem.

        cout << "Pass from: " << offRecv[turn] << " to: " << offRecv[turn]+passpt << endl;
        passSwept(state[0], state[2], tmine);
        cudaCheckError(cudaMemcpy(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(dState + xgp, state[2] + offSend[turn], passsize, cudaMemcpyHostToDevice));
        swIncrement();//Increment
        cudaDeviceSynchronize();

        if(error != cudaSuccess)
        {
            printf("CUDA error UPPASS: %s\n", cudaGetErrorString(error));
        }

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

        for (int k=0; k<(cGlob.cBks-1); k++)
        {
            ix = 2*(k/cmid);
            if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
            {
                splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
            }
            else
            {
                wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
            }
        }
        
        cudaCheckError(cudaMemcpy(state[1], dState, gpusize, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error SPLIT: %s\n", cudaGetErrorString(error));
        }

        // ------------ Pass Edges ------------ //

        passSwept(state[2], state[0], tmine);
        cudaCheckError(cudaMemcpy(state[2] + offRecv[turn], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(dState, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice));
        swIncrement();

        // Increment Counter and timestep
        tmine += cGlob.tpb;
        t_eq += cGlob.dt * (tmine/NSTEPS);

        while(t_eq < cGlob.tf)
        {
            // ------------ Step Forward ------------ //
            // ------------ WHOLE ------------ //

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

            for (int k=0; k<cGlob.cBks; k++)
            {
                ix = 2*(k/cmid);
                wholeDiamondCPU(state[ix] + k*cGlob.tpb, tmine);
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

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);
            
            for (int k=0; k<(cGlob.cBks-1); k++)
            {
                ix = 2*(k/cmid);
                if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
                {
                    splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
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
                downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

                for (int k=0; k<cGlob.cBks; k++)
                {
                    ix = 2*(k/cmid);
                    downTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
                }

                cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);
                                
                for (int i=0; i<3; i++)
                {
                    for (int k=1; k<=alen[i]; k++)  solutionOutput(state[i], t_eq, k, xpts[i]);
                }  

                upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

                for (int k=0; k<cGlob.cBks; k++)
                { 
                    ix = 2*(k/cmid);
                    upTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
                }

                passSwept(state[0], state[2], tmine);
                cudaMemcpyAsync(state[0] + offRecv[turn], dState + 1, passsize, cudaMemcpyDeviceToHost, st1);
                cudaMemcpyAsync(dState + xgp, state[0] + offSend[turn], passsize, cudaMemcpyHostToDevice, st2);
                swIncrement();//Increment

                // Increment Counter and timestep
                tmine += cGlob.tpb;
                t_eq += cGlob.dt * (tmine/NSTEPS);

                // ------------ Step Forward ------------ //
                // ------------ SPLIT ------------ //

                wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);
                
                for (int k=0; k<(cGlob.cBks-1); k++)
                {
                    ix = 2*(k/cmid);
                    if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
                    {
                        splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
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
        
        for (int k=0; k<cGlob.cBks; k++)
        { 
            upTriangleCPU(state[0] + k*cGlob.tpb, tmine);
        }
        passSwept(state[0], state[0], tmine); // Left first
        swIncrement();

        for (int k=0; k<(cGlob.cBks-1); k++)
        {
            if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
            {
                splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
            }
            else
            {
                wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
            }
        }
        tmine += cGlob.tpb;
        t_eq += cGlob.dt * (tmine/NSTEPS);
        passSwept(state[0], state[0], tmine); 
        swIncrement();

        while (t_eq < cGlob.tf)
        {

            for (int k=0; k<cGlob.cBks; k++)
            {
                wholeDiamondCPU(state[0] + k*cGlob.tpb, tmine);
            }

            passSwept(state[0], state[0], tmine); 
            swIncrement();

            for (int k=0; k<(cGlob.cBks-1); k++)
            {
                if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
                {
                    splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                }
            }

            passSwept(state[0], state[0], tmine); 
            swIncrement();

            if (t_eq > twrite)
            {
                for (int k=0; k<cGlob.cBks; k++)
                {
                    downTriangleCPU(state[0] + k*cGlob.tpb, tmine);
                }

                for (int k=1; k<=cGlob.xcpu; k++)  solutionOutput(state[0], t_eq, k, xpts[0]);

                for (int k=0; k<cGlob.cBks; k++)
                { 
                    upTriangleCPU(state[0] + k*cGlob.tpb, tmine);
                }

                passSwept(state[0], state[0], tmine);
                swIncrement();//Increment

                // Increment Counter and timestep
                tmine += cGlob.tpb;
                t_eq += cGlob.dt * (tmine/NSTEPS);

                // ------------ Step Forward ------------ //
                // ------------ SPLIT ------------ //
                
                for (int k=0; k<(cGlob.cBks-1); k++)
                {
                    if ((ranks[1] == lastproc) && (k ==(cGlob.cBks-1)))
                    {
                        splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                }

                passSwept(state[0], state[0], tmine);
                swIncrement();//Increment
                // Increment Counter and timestep
                tmine += cGlob.tpb;
                t_eq += cGlob.dt * (tmine/NSTEPS);
                twrite += cGlob.freq;
            }
        }
    }
    *tstep = tmine;
    return t_eq;
}
