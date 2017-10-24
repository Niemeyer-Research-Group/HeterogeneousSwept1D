
/*
---------------------------
    SWEPT CORE
---------------------------
*/

// SET HBOUNDS!

using namespace std;

// int offSend[2];
// int offRecv[2];
// int cnt, turn;

// void swIncrement()
// {
//     cnt++;
//     turn = cnt & 1;
// }
typedef std::vector<int> ivec;

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
    int tnow=tstep;
    for (int k=2; k<cGlob.htp; k++)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            stepUpdate(state, n, tnow);
        }
    tnow++;
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
    states ssLeft[3]; 
    ssLeft[2] = bound[0]; 
    states ssRight[3]; 
    ssRight[0] = bound[1]; 
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n == cGlob.ht)
            {
                ssLeft[0] = state[n-1], ssLeft[1] = state[n];
                stepUpdate(ssLeft, n, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(ssRight, n, tnow);
                state[n] = ssRight[1];
            }
            else
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
            if (n == cGlob.ht)
            {
                ssLeft[0] = state[n-1], ssLeft[1] = state[n];
                stepUpdate(ssLeft, n, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(ssRight, n, tnow);
                state[n] = ssRight[1];
            }
            else
            {
                stepUpdate(state, n, tnow);
            }
        }
	tnow++;
    }
}

void passSwept(states *putst, states *getst, int tstep, int turn)
{
    int rx = turn^1;
    cout << "Entered Pass " << ranks[1] << endl;
    MPI_Isend(putst, cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD, &req[0]);

    MPI_Recv(getst, cGlob.htp, struct_type, ranks[2*rx], TAGS(tstep),
            MPI_COMM_WORLD,  MPI_STATUS_IGNORE);

    cout << "Exit Pass " << ranks[1] << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request_free(&req[0]);
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


double sweptWrapper(states **state, ivec xpts, ivec alen, int *tstep)
{
    if (!ranks[1]) std::cout << "Swept Decomposition" << std::endl;
    int tmine = *tstep;
    const int bkL = cGlob.cBks - 1;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    states putst[cGlob.htp], getst[cGlob.htp];

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        const int xc = cGlob.xcpu/2, xcp=xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int cmid = cGlob.cBks/2;
        int ix;

        const size_t gpusize = cGlob.szState * (xgpp + cGlob.ht);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;

        int gpupts = gpusize/cGlob.szState;

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);

        std::cout << "GPU threads launched: " << cGlob.tpb*cGlob.gBks << " GpuSize: " << gpupts << "  " << passsize/cGlob.szState << " node ht: " << cGlob.ht << " | sizeof states: " << cGlob.szState << std::endl;
        
        cout << "AFTER Initial copy " << endl;

        states *dState;

        cudaCheckError(cudaMalloc((void **)&dState, gpusize));

        cudaCheckError(cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice));

        /* 
        -- DOWN MUST FOLLOW A SPLIT AND UP CANNOT BE IN WHILE LOOP SO DO UP AND FIRST SPLIT OUTSIDE OF LOOP THEN LOOP CAN BE WITH WHOLE - DIAMOND - CHECK DOWN.
        */
        // ------------ Step Forward ------------ //
        // ------------ UP ------------ //

        upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

        for (int k=0; k<cGlob.cBks; k++)
        { 
            ix = 2*(k/cmid);
            upTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
        }

        cout << "After UPTRIANGLE " << endl;

        // ------------ Pass Edges ------------ // 
        // -- FRONT TO BACK -- //

        cudaCheckError(cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice));

        for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
        passSwept(&putst[0], &getst[0], tmine, 0);
        for (int k=0; k<cGlob.htp; k++) getst[k] = state[2][k + xcp];

        //swIncrement();//Increment

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //

        wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

        for (int k=0; k<(cGlob.cBks); k++)
        {
            ix = 2*(k/cmid);
            if ((ranks[1] == lastproc) && (k == bkL))
            {
                splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                cout << "Yes Split Diamond occurred " << ranks[1] << " " << k << endl;
            }
            else
            {
                wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
            }
        }      

        // ------------ Pass Edges ------------ //
        // -- BACK TO FRONT -- //

        cudaCheckError(cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice));

        for (int k=0; k<cGlob.htp; k++) putst[k] = state[2][k+xc];
        passSwept(&putst[0], &getst[0], tmine, 1);
        for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];

        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);

        if (!ranks[1]) state[0][0] = bound[0];
        if (ranks[1]==lastproc) state[2][xcp] = bound[1];

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
            
            // ------------ Pass Edges ------------ //
            
            cudaCheckError(cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice));
    
            for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
            passSwept(&putst[0], &getst[0], tmine, 0);
            for (int k=0; k<cGlob.htp; k++) getst[k] = state[2][k + xcp];

            // Increment Counter and timestep
            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);

            // ------------ Step Forward ------------ //
            // ------------ SPLIT ------------ //

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);
            
            for (int k=0; k<(cGlob.cBks); k++)
            {
                ix = 2*(k/cmid);
                if ((ranks[1] == lastproc) && (k == bkL))
                {
                    splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                }
            }

            // ------------ Pass Edges ------------ //
            // -- BACK TO FRONT -- //

            cudaCheckError(cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice));

            for (int k=0; k<cGlob.htp; k++) putst[k] = state[2][k+xc];
            passSwept(&putst[0], &getst[0], tmine, 1);
            for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];

            // Increment Counter and timestep
            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);

            if (!ranks[1]) state[0][0] = bound[0];
            if (ranks[1]==lastproc) state[2][xcp] = bound[1];

            if (t_eq > twrite)
            {
                downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

                for (int k=0; k<cGlob.cBks; k++)
                {
                    ix = 2*(k/cmid);
                    downTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
                }
                
                // Increment Counter and timestep
                tmine += cGlob.ht;
                t_eq = cGlob.dt * (tmine/NSTEPS);

                cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);
                                
                for (int i=0; i<3; i++)
                {
                    for (int k=1; k<alen[i]; k++)  solutionOutput(state[i], t_eq, k, xpts[i]);
                }  

                upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

                for (int k=0; k<cGlob.cBks; k++)
                { 
                    ix = 2*(k/cmid);
                    upTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
                }

                cudaCheckError(cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost));
                cudaCheckError(cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice));
        
                for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
                passSwept(&putst[0], &getst[0], tmine, 0);
                for (int k=0; k<cGlob.htp; k++) getst[k] = state[2][k + xcp];

                // ------------ Step Forward ------------ //
                // ------------ SPLIT ------------ //

                wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);
                
                for (int k=0; k<(cGlob.cBks); k++)
                {
                    ix = 2*(k/cmid);
                    if ((ranks[1] == lastproc) && (k ==bkL))
                    {
                        splitDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(state[ix] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                }

                // ------------ Pass Edges ------------ //
                // -- BACK TO FRONT -- //

                cudaCheckError(cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost));
                cudaCheckError(cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice));

                for (int k=0; k<cGlob.htp; k++) putst[k] = state[2][k+xc];
                passSwept(&putst[0], &getst[0], tmine+1, 1);
                for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];

                // Increment Counter and timestep
                tmine += cGlob.ht;
                t_eq = cGlob.dt * (tmine/NSTEPS);
                twrite += cGlob.freq;
                if (!ranks[1]) state[0][0] = bound[0];
                if (ranks[1]==lastproc) state[2][xcp] = bound[1];
            }
            if (tmine % 1000 == 0) std::cout << t_eq << " | " << cGlob.tf << std::endl;
        }

        downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);
        
        for (int k=0; k<cGlob.cBks; k++)
        {
            ix = 2*(k/cmid);
            downTriangleCPU(state[ix] + k*cGlob.tpb, tmine);
        }
        
        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);

        cudaMemcpy(state[1], dState, ptsize, cudaMemcpyDeviceToHost);

        cudaFree(dState);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
    }

    else
    {
        const int xc = cGlob.xcpu, xcp = xc+1;
        for (int k=0; k<cGlob.cBks; k++)
        { 
            upTriangleCPU(state[0] + k*cGlob.tpb, tmine);
        }

        for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
        passSwept(&putst[0], &getst[0], tmine, 0);
        for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k+xcp];

        //Split
        for (int k=0; k<(cGlob.cBks); k++)
        {
            if ((ranks[1] == lastproc) && (k == bkL))
            {
                splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
            }
            else
            {
                wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
            }
        }

        // -- BACK TO FRONT -- //

        for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+xc];
        passSwept(&putst[0], &getst[0], tmine+1, 1);
        for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];

        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);
        if (!ranks[1]) state[0][0] = bound[0];
        if (ranks[1]==lastproc) state[0][xcp] = bound[1];

        while (t_eq < cGlob.tf)
        {
            for (int k=0; k<cGlob.cBks; k++)
            {
                wholeDiamondCPU(state[0] + k*cGlob.tpb, tmine);
            }

            for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
            passSwept(&putst[0], &getst[0], tmine, 0);
            for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k + xcp];

            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);

            for (int k=0; k<(cGlob.cBks); k++)
            {
                if ((ranks[1] == lastproc) && (k == bkL))
                {
                    splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                }
            }

            for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+xc];
            passSwept(&putst[0], &getst[0], tmine, 1);
            for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];
    
            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);
            if (!ranks[1]) state[0][0] = bound[0];
            if (ranks[1]==lastproc) state[0][xcp] = bound[1];

            if (t_eq > twrite)
            {
                for (int k=0; k<cGlob.cBks; k++)
                {
                    downTriangleCPU(state[0] + k*cGlob.tpb, tmine);
                }

                // Increment Counter and timestep
                tmine += cGlob.tpb;
                t_eq = cGlob.dt * (tmine/NSTEPS);

                for (int k=1; k<alen[0]; k++)  solutionOutput(state[0], t_eq, k, xpts[0]);

                for (int k=0; k<cGlob.cBks; k++)
                { 
                    upTriangleCPU(state[0] + k*cGlob.tpb, tmine);
                }

                for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+1];
                passSwept(&putst[0], &getst[0], tmine, 0);
                for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k + xcp];

                // ------------ Step Forward ------------ //
                // ------------ SPLIT ------------ //
                
                for (int k=0; k<(cGlob.cBks); k++)
                {
                    if ((ranks[1] == lastproc) && (k == bkL))
                    {
                        splitDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(state[0] + cGlob.ht + k*cGlob.tpb, tmine);
                    }
                }

                for (int k=0; k<cGlob.htp; k++) putst[k] = state[0][k+xc];
                passSwept(&putst[0], &getst[0], tmine+1, 1);
                for (int k=0; k<cGlob.htp; k++) getst[k] = state[0][k];
        
                // Increment Counter and timestep
                tmine += cGlob.tpb;
                t_eq = cGlob.dt * (tmine/NSTEPS);
                twrite += cGlob.freq;
                if (!ranks[1]) state[0][0] = bound[0];
                if (ranks[1]==lastproc) state[0][xcp] = bound[1];
            }
        }
        for (int k=0; k<cGlob.cBks; k++)
        {
            downTriangleCPU(state[0] + k*cGlob.tpb, tmine);
        }
    }
    *tstep = tmine;
    return t_eq;
}
