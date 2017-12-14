
/*
---------------------------
    SWEPT CORE
---------------------------
*/

using namespace std;

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
downTriangle(states *state, const int tstep, const int offset)
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
wholeDiamond(states *state, const int tstep, const int offset)
{
	extern __shared__ states temper[];

    int tidx = threadIdx.x + 1; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + threadIdx.x + offset;

    int tnow = tstep;
	int k;
    if (threadIdx.x<2) temper[threadIdx.x] = state[gid];
    __syncthreads();
    temper[tidx+1] = state[gid + 2];

    __syncthreads();

	for (k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
        	stepUpdate(temper, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}


	for (k=2; k<=mid; k++)
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
                stepUpdate(&ssLeft[0], n, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], n, tnow);
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
                stepUpdate(&ssLeft[0], n, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], n, tnow);
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

void passSwept(states *passer, states *getter, const int tstep, const int turn)
{
    int rx = turn^1;
    int t0;
    MPI_Isend(passer, cGlob.htp, struct_type, ranks[2*turn], TAGS(tstep),
            MPI_COMM_WORLD, &req[0]);

    MPI_Recv(getter, cGlob.htp, struct_type, ranks[2*rx], TAGS(tstep),
            MPI_COMM_WORLD, &stat[0]);

    MPI_Wait(&req[0], &stat[0]);
}

// void applyBC(states *state, int ty, int pt)
// {
//     // Like if-dirichilet
//     // Works for whole
//     state[ty*pt] = sBound[ty];
//     // If reflective
//     // state[ty*pt] = state[pt-2] or state[pt+2]
// }


// Now we need to put the last value in a bucket, and append that to the start of the next array.
double sweptWrapper(states **state, const ivec xpts, const ivec alen, int *tstep)
{
	if (!ranks[1]) cout << "SWEPT Decomposition" << endl;
    int tmine = *tstep;
    int t0;
    const int bkL = cGlob.cBks - 1;

    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;

    // Must be declared global in equation specific header.
    stPass = cGlob.htp;
    numPass = NSTATES * stPass;

    int tou = 2000;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        const int xc = cGlob.xcpu/2, xcp=xc+1;
        const int xgp = cGlob.xg+1, xgpp = cGlob.xg+2;
        const int cmid = cGlob.cBks/2;
        int bx, ix;

        const size_t gpusize = cGlob.szState * (xgpp + cGlob.ht);
        const size_t ptsize = cGlob.szState * xgpp;
        const size_t passsize =  cGlob.szState * cGlob.htp;
        const size_t smem = cGlob.szState * cGlob.base;

        int gpupts = gpusize/cGlob.szState;

        cudaStream_t st1, st2;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);

        states *dState;

        cudaMalloc((void **)&dState, gpusize);
        cudaMemcpy(dState, state[1], ptsize, cudaMemcpyHostToDevice);

        /*
        -- DOWN MUST FOLLOW A SPLIT AND UP CANNOT BE IN WHILE LOOP SO DO UP AND FIRST SPLIT OUTSIDE OF LOOP THEN LOOP CAN BE WITH WHOLE - DIAMOND - CHECK DOWN.
        */
        // ------------ Step Forward ------------ //
        // ------------ UP ------------ //

        upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

        for (int k=0; k<cGlob.cBks; k++)
        {
            bx = (k/cmid);
            ix = 2*bx;
            upTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
        }

        // ------------ Pass Edges ------------ //
        // -- FRONT TO BACK -- //

        cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

        // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
        // unstructify(&putSt[0], &putRe[0]);

        // passSwept(&putRe[0], &getRe[0], tmine, 0);
        passSwept(state[0] + 1, state[2] + xcp, tmine, 0);

        // restructify(&getSt[0], &getRe[0]);
        // for (int k=0; k<cGlob.htp; k++) state[2][k+xcp] = getSt[k];

        // ------------ Step Forward ------------ //
        // ------------ SPLIT ------------ //
        // cout << "SPLIT CALL" << endl;

        wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

        for (int k=0; k<(cGlob.cBks); k++)
        {
            bx = (k/cmid);
            ix = 2*bx;
            if ((ranks[1] == lastproc) && (k == bkL))
            {
                splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
            }
            else
            {
                wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
            }
        }

        // cout << "SPLIT " << endl;

        // ------------ Pass Edges ------------ //
        // -- BACK TO FRONT -- //

        cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

        // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[2][k+xc];
        // unstructify(&putSt[0], &putRe[0]);

        // passSwept(&putRe[0], &getRe[0], tmine+1, 1);
        passSwept(state[2] + xc, state[0], tmine+1, 1);

        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);

        if (!ranks[1]) state[0][0] = bound[0];
        if (ranks[1]==lastproc) state[2][xcp] = bound[1];
        // cout << "WHILE " << endl;

        while(t_eq < cGlob.tf)
        {
            // ------------ Step Forward ------------ //
            // ------------ WHOLE ------------ //

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

            for (int k=0; k<cGlob.cBks; k++)
            {
                bx = (k/cmid);
                ix = 2*(k/cmid);
                wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
            }

            // ------------ Pass Edges ------------ //

            cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
            cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

            // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
            // unstructify(&putSt[0], &putRe[0]);

            // passSwept(&putRe[0], &getRe[0], tmine, 0);
            passSwept(state[0] + 1, state[2] + xcp, tmine, 0);

            // restructify(&getSt[0], &getRe[0]);
            // for (int k=0; k<cGlob.htp; k++) state[2][k+xcp] = getSt[k];

            // Increment Counter and timestep
            tmine += cGlob.ht;
            t_eq = cGlob.dt * (tmine/NSTEPS);

            // ------------ Step Forward ------------ //
            // ------------ SPLIT ------------ //

            wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

            for (int k=0; k<(cGlob.cBks); k++)
            {
                bx = (k/cmid);
                ix = 2*bx;
                if ((ranks[1] == lastproc) && (k == bkL))
                {
                    splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                }
                else
                {
                    wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                }
            }

            // ------------ Pass Edges ------------ //
            // -- BACK TO FRONT -- //

            cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
            cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

            // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[2][k+xc];
            // unstructify(&putSt[0], &putRe[0]);

            // passSwept(&putRe[0], &getRe[0], tmine, 1);
            passSwept(state[2] + xc, state[0], tmine, 1);

            // restructify(&getSt[0], &getRe[0]);
            // for (int k=0; k<cGlob.htp; k++)  state[0][k] = getSt[k];

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
                    bx = (k/cmid);
                    ix = 2*bx;
                    downTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
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
                    bx = (k/cmid);
                    ix = 2*bx;
                    upTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
                }

                cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
                cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

                // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
                // unstructify(&putSt[0], &putRe[0]);

                // passSwept(&putRe[0], &getRe[0], tmine, 0);
                passSwept(state[0] + 1, state[2] + xcp, tmine, 0);

                // restructify(&getSt[0], &getRe[0]);
                // for (int k=0; k<cGlob.htp; k++) state[2][k + xcp] = getSt[k];

                // ------------ Step Forward ------------ //
                // ------------ SPLIT ------------ //

                wholeDiamond <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, cGlob.ht);

                for (int k=0; k<(cGlob.cBks); k++)
                {
                    bx = (k/cmid);
                    ix = 2*bx;
                    if ((ranks[1] == lastproc) && (k ==bkL))
                    {
                        splitDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                    }
                    else
                    {
                        wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb + cGlob.ht, tmine);
                    }
                }

                // ------------ Pass Edges ------------ //
                // -- BACK TO FRONT -- //
                cudaMemcpy(state[2], dState+cGlob.xg, passsize, cudaMemcpyDeviceToHost);
                cudaMemcpy(dState, state[0] + xc, passsize, cudaMemcpyHostToDevice);

                // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[2][k+xc];
                // unstructify(&putSt[0], &putRe[0]);

                // passSwept(&putRe[0], &getRe[0], tmine, 1);
                passSwept(state[2] + xc, state[0], tmine+1, 1);

                // restructify(&getSt[0], &getRe[0]);
                // for (int k=0; k<cGlob.htp; k++)  state[0][k] = getSt[k];

                // Increment Counter and timestep
                tmine += cGlob.ht;
                t_eq = cGlob.dt * (tmine/NSTEPS);
                twrite += cGlob.freq;
                if (!ranks[1]) state[0][0] = bound[0];
                if (ranks[1]==lastproc) state[2][xcp] = bound[1];
            }

        }

        downTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine, 0);

        for (int k=0; k<cGlob.cBks; k++)
        {
            bx = (k/cmid);
            ix = 2*bx;
            downTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
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

        // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
        // unstructify(&putSt[0], &putRe[0]);

        // passSwept(&putRe[0], &getRe[0], tmine, 0);
        passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

        // restructify(&getSt[0], &getRe[0]);
        // for (int k=0; k<cGlob.htp; k++) state[0][k + xcp] = getSt[k];

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

        // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+xc];
        // unstructify(&putSt[0], &putRe[0]);

        // passSwept(&putRe[0], &getRe[0], tmine+1, 1);
        passSwept(state[0] + xc, state[0], tmine+1, 1);

        // restructify(&getSt[0], &getRe[0]);
        // for (int k=0; k<cGlob.htp; k++)  state[0][k] = getSt[k];

        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);
        if (!ranks[1]) state[0][0] = bound[0];
        if (ranks[1]==lastproc) state[0][xcp] = bound[1];

        while (t_eq < cGlob.tf)
        {
			#ifdef PULSE
				if (!tmine/cGlob.tpb && !ranks[0]) cout << "It's alive: " << tmine << endl;
			#endif

            for (int k=0; k<cGlob.cBks; k++)
            {
                wholeDiamondCPU(state[0] + k*cGlob.tpb, tmine);
            }

            // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
            // unstructify(&putSt[0], &putRe[0]);

            // passSwept(&putRe[0], &getRe[0], tmine, 0);
            passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

            // restructify(&getSt[0], &getRe[0]);
            // for (int k=0; k<cGlob.htp; k++) state[0][k+xcp] = getSt[k];

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

            // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+xc];
            // unstructify(&putSt[0], &putRe[0]);

            // passSwept(&putRe[0], &getRe[0], tmine, 1);
            passSwept(state[0] + xc, state[0], tmine, 1);

            // restructify(&getSt[0], &getRe[0]);
            // for (int k=0; k<cGlob.htp; k++)  state[0][k] = getSt[k];

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

                // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+1];
                // unstructify(&putSt[0], &putRe[0]);

                // passSwept(&putRe[0], &getRe[0], tmine, 0);
                passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

                // restructify(&getSt[0], &getRe[0]);
                // for (int k=0; k<cGlob.htp; k++) state[0][k + xcp] = getSt[k];

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

                // for (int k=0; k<cGlob.htp; k++) putSt[k] = state[0][k+xc];
                // unstructify(&putSt[0], &putRe[0]);

                // passSwept(&putRe[0], &getRe[0], tmine+1, 1);
                passSwept(state[0] + xc, state[0], tmine+1, 1);

                // restructify(&getSt[0], &getRe[0]);
                // for (int k=0; k<cGlob.htp; k++)  state[0][k] = putSt[k];

                // Increment Counter and timestep
                tmine += cGlob.ht;
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
        // Increment Counter and timestep
        tmine += cGlob.ht;
        t_eq = cGlob.dt * (tmine/NSTEPS);
    }

    *tstep = tmine;
    return t_eq;
}
