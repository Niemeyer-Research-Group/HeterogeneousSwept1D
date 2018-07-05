
/*
---------------------------
    SWEPT CORE
---------------------------
*/

using namespace std;

typedef std::vector<int> ivec;

states ssLeft[3];
states ssRight[3];

__global__ void upTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedstate[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;

    // Using tidx as tid is kind of confusing for reader but looks valid.

	sharedstate[tidx] = state[gid + 1];

    __syncthreads();

	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepUpdate(sharedstate, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid + 1] = sharedstate[tidx];
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
	extern __shared__ states sharedstate[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + tid + offset;
    int tidx = tid + 1;

    int tnow = tstep; // read tstep into register.

    if (tid<2) sharedstate[tid] = state[gid];
    __syncthreads();
    sharedstate[tid+2] = state[gid+2];
    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
	    	stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}
    state[gid] = sharedstate[tidx];
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
	extern __shared__ states sharedstate[];

    int tidx = threadIdx.x + 1; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2;
    int gid = blockDim.x * blockIdx.x + threadIdx.x + offset;

    int tnow = tstep;
	int k;
    if (threadIdx.x<2) sharedstate[threadIdx.x] = state[gid];
    __syncthreads();
    sharedstate[tidx+1] = state[gid + 2];

    __syncthreads();

	for (k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
        	stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}

	for (k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepUpdate(sharedstate, tidx, tnow);
		}
		tnow++;
		__syncthreads();
    }
    state[gid + 1] = sharedstate[tidx];
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

void wholeDiamondCPU(states *state, int tnow)
{
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

void splitDiamondCPU(states *state, int tnow)
{
    ssLeft[2] = bound[1];
    ssRight[0] = bound[0];
    for (int k=cGlob.ht; k>0; k--)
    {
        for (int n=k; n<(cGlob.base-k); n++)
        {
            if (n == cGlob.ht)
            {
                ssLeft[0] = state[n-1], ssLeft[1] = state[n];
                stepUpdate(&ssLeft[0], 1, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], 1, tnow);
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
                stepUpdate(&ssLeft[0], 1, tnow);
                state[n] = ssLeft[1];
            }
            else if (n == cGlob.htp)
            {
                ssRight[1] = state[n], ssRight[2] = state[n+1];
                stepUpdate(&ssRight[0], 1, tnow);
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

void passSwept(states *passer, states *getter, int tstep, int turn)
{
    int rx = turn^1;

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
double sweptWrapper(states **state,  int *tstep)
{
	// FILE *diagDump;
	// std::string fname = "edge/edgeWrite_" + std::to_string(ranks[1]) + ".csv";
	// diagDump = fopen(fname.c_str(), "w+");
    
	if (!ranks[1]) cout << "SWEPT Decomposition " << cGlob.tpb << endl;
    const int bkL = cGlob.cBks - 1;
    double t_eq = 0.0;
    double twrite = cGlob.freq - QUARTER*cGlob.dt;
    int tmine = *tstep;

    // Must be declared global in equation specific header.

    int tou = 2000;

    if (cGlob.hasGpu) // If there's no gpu assigned to the process this is 0.
    {
        cout << ranks[1] << " hasGpu" << endl;
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

        /* RULES
        -- DOWN MUST FOLLOW A SPLIT 
        -- UP CANNOT BE IN WHILE LOOP 
        -||- ACTION: DO UP AND FIRST SPLIT OUTSIDE OF LOOP 
        -||- THEN LOOP CAN BEGIN WITH WHOLE DIAMOND.
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

        passSwept(state[0] + 1, state[2] + xcp, tmine, 0);

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

        passSwept(state[2] + xc, state[0], tmine+1, 1);

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
                bx = (k/cmid);
                ix = 2*(k/cmid);
                wholeDiamondCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
            }

            // ------------ Pass Edges ------------ //

            cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
            cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

            passSwept(state[0] + 1, state[2] + xcp, tmine, 0);


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

            passSwept(state[2] + xc, state[0], tmine, 1);

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

                writeOut(state, t_eq);
                
                upTriangle <<<cGlob.gBks, cGlob.tpb, smem>>> (dState, tmine);

                for (int k=0; k<cGlob.cBks; k++)
                {
                    bx = (k/cmid);
                    ix = 2*bx;
                    upTriangleCPU(state[ix] + (k - bx * cmid)*cGlob.tpb, tmine);
                }

                cudaMemcpy(state[0] + xcp, dState + 1, passsize, cudaMemcpyDeviceToHost);
                cudaMemcpy(dState + xgp, state[2] + 1, passsize, cudaMemcpyHostToDevice);

                passSwept(state[0] + 1, state[2] + xcp, tmine, 0);


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

                passSwept(state[2] + xc, state[0], tmine+1, 1);

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

        // for (int i=0; i<xcp; i++) fprintf(diagDump,"%.5f:%.5f,", state[0][i].T[0], state[0][i].T[1]);
        // fprintf(diagDump, "\n");

        passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

        //Split
        for (int k=0; k<cGlob.cBks; k++)
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

        // for (int i=cGlob.ht; i<xcp+cGlob.ht; i++) fprintf(diagDump,"%.5f:%.5f,", state[0][i].T[0], state[0][i].T[1]);
        // fprintf(diagDump, "\n");
        // -- BACK TO FRONT -- //

        passSwept(state[0] + xc, state[0], tmine+1, 1);

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

			// for (int i=1; i<xcp; i++) fprintf(diagDump,"%.5f:%.5f,", state[0][i].T[0], state[0][i].T[1]);
            // fprintf(diagDump, "\n");

            passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

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

			// for (int i=cGlob.ht; i<xcp+cGlob.ht; i++) fprintf(diagDump,"%.5f:%.5f,", state[0][i].T[0], state[0][i].T[1]);
			// fprintf(diagDump, "\n");

            passSwept(state[0] + xc, state[0], tmine, 1);

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
                tmine += cGlob.ht;
                t_eq = cGlob.dt * (tmine/NSTEPS);

                writeOut(state, t_eq);

                for (int k=0; k<cGlob.cBks; k++)
                {
                    upTriangleCPU(state[0] + k*cGlob.tpb, tmine);
                }

                passSwept(state[0] + 1, state[0] + xcp, tmine, 0);

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

                // passSwept(&putRe[0], &getRe[0], tmine+1, 1);
                passSwept(state[0] + xc, state[0], tmine+1, 1);

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
	
	// if (!ranks[1]){
	// 	FILE *metaFile;
	// 	metaFile = fopen("edge/RunDetails.txt", "w+");
	// 	fprintf(metaFile, "%d,%.5f,%.5f,%.5f\n", cGlob.tpb, cGlob.lx, cGlob.dx, cGlob.dt);
	// 	fclose(metaFile);
	// }


	// fclose(diagDump);
    *tstep = tmine;
    //atomicWrite(timeit.typ, timeit.times);
    return t_eq;
}
