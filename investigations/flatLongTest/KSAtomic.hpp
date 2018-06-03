


struct states {
   double u[2]; // Full Step, Midpoint step state variables
   double uxx;
};

__device__ __forceinline__
double sumu(states ust, const int ix) { return ust.u[ix] + ust.uxx; }


__device__
void ksStep(states *state, const int idx[3], const int tstep)
{
    const int itx = (tstep ^ 1);
    state[idx[1]].u[tstep] = state[idx[1]].u[0] - (HALF * disc.dt * (itx+1)) *
                        (convect(state[idx[0]].u[itx], state[idx[2]].u[itx]) +
                        secondDer(sumu(state[idx[0]], itx), sumu(state[idx[2]], itx), sumu(state[idx[1]], itx)));
}

__device__
void stepUpdate(states *state, const int idx[3], const int tstep)
{
    const int ts = DIVMOD(tstep);
    if (tstep & 1) //Odd - Rslt is 0 for even numbers
    {
        state[idx[1]].uxx = secondDer(state[idx[0]].u[ts],
                                    state[idx[2]].u[ts],
                                    state[idx[1]].u[ts]);
    }
    else
    {
        ksStep(state, idx, ts);
    }
}


__global__ void classicStep(states *state, const int ts)
{
	auto gidz = idxes<3>();

    stepUpdate(state, gidz.a, ts);
}

__global__ void upTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedState[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int mid = blockDim.x >> 1;
    const int tidx[3] = {threadIdx.x-1, threadIdx.x, threadIdx.x+1};

    // Using tidx as tid is kind of confusing for reader but looks valid.

	sharedState[tidx[1]] = state[gid];

    __syncthreads();

	for (int k=1; k<mid; k++)
	{
		if (tidx[1] < (blockDim.x-k) && tidx[1] >= k)
		{
            stepUpdate(sharedState, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid] = sharedState[tidx[1]];
}


__global__
void
downTriangle(states *state, const int tstep)
{
	extern __shared__ states sharedState[];

    const int tid = threadIdx.x; // Thread index
    const int mid = blockDim.x >> 1; // Half of block size
    const int base = blockDim.x + 2;
    const int gid = blockDim.x * blockIdx.x + tid;
    const int tidx[3] = {tid, tid + 1, tid + 2};

    int tnow = tstep; // read tstep into register.

    if (tid<2) sharedState[tid] = state[disc.modg(gid-1)];
    sharedState[tid+2] = state[disc.modg(gid+1)];
    __syncthreads();

	for (int k=mid; k>0; k--)
	{
		if (tidx[1] < (base-k) && tidx[1] >= k)
		{
	    	stepUpdate(sharedState, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}
    state[gid] = sharedState[tidx[1]];
}

template <bool SPLIT>
__global__ void
wholeDiamond(states *state, const int tstep)
{
	extern __shared__ states sharedState[];

    const int mid = (blockDim.x >> 1); // Half of block size
    const int base = blockDim.x + 2;
    const int gid = blockDim.x * blockIdx.x + threadIdx.x + SPLIT*(disc.ht-2);
    const int tidx[3] = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2};

    int tnow = tstep;
	int k;

    if (threadIdx.x<2) sharedState[threadIdx.x] = state[disc.modg(gid-1)];
    sharedState[tidx[2]] = state[disc.modg(gid+1)];

    __syncthreads();

	for (k=mid; k>0; k--)
	{
		if (tidx[1] < (base-k) && tidx[1] >= k)
		{
        	stepUpdate(sharedState, tidx, tnow);
		}
		tnow++;
		__syncthreads();
	}

	for (k=2; k<=mid; k++)
	{
		if (tidx[1] < (base-k) && tidx[1] >= k)
		{
            stepUpdate(sharedState, tidx, tnow);
		}
		tnow++;
		__syncthreads();
    }

    state[disc.modg(gid)] = sharedState[tidx[1]];
}



double classicWrapper(states *aState)
{
    std::cout << " -- Classic Atom -- " << std::endl;
    const size_t szSt = sizeof(states); 
    int tstep = 1;
    const int npass = rv.dv * szSt;

    states *dState;

    cudaMalloc((void **) &dState, npass);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dState, aState, npass, cudaMemcpyHostToDevice);

    double t_eq = 0.0;

    while (t_eq <= rv.tf)
    {
        classicStep <<< rv.bks, rv.tpb >>> (dState, tstep);
        tstep++;
        if (tstep%NSTEPS == 0) t_eq += rv.dt;
    }

    cudaMemcpy(aState, dState, npass, cudaMemcpyDeviceToHost);
    std::cout << tstep << std::endl;
    cudaFree(dState);

    return t_eq;
}

//The host routine.
double
sweptWrapper(states *aState)
{
    std::cout << " -- Swept Atom -- " << std::endl;
    const size_t szSt = sizeof(states); 
    const size_t smem = (rv.tpb+2)*szSt;
    
    std::cout << "SHARED MEMORY: " << smem << std::endl;
    int tstep = 1;
	states *dState;
    const int npass = rv.dv * szSt;

	cudaMalloc((void **)&dState, npass);

	// Copy the initial conditions to the device array.
	cudaMemcpy(dState, aState, npass, cudaMemcpyHostToDevice);
	//Start the counter and start the clock.

	//Every other step is a full timestep and each cycle is half tpb steps.
    
    int stepi = rv.tpb/2;
	double t_fullstep = 0.25 * rv.dt * (double)rv.tpb;
	double t_eq;

	t_eq = t_fullstep;

	upTriangle <<< rv.bks, rv.tpb, smem >>> (dState, tstep);

	//Split
	wholeDiamond <true> <<< rv.bks, rv.tpb, smem >>> (dState, tstep);

    tstep += stepi;

	// Call the kernels until you reach the iteration limit.
	while(t_eq < rv.tf)
	{
		wholeDiamond <false> <<< rv.bks, rv.tpb, smem >>> (dState, tstep);

        tstep += stepi;
		
		wholeDiamond <true> <<< rv.bks, rv.tpb, smem >>> (dState, tstep);

		t_eq  += t_fullstep;
        tstep += stepi;
	}

	downTriangle <<< rv.bks, rv.tpb, smem >>>(dState, tstep);

    tstep += stepi;

	cudaMemcpy(aState, dState, npass, cudaMemcpyDeviceToHost);

	cudaFree(dState);
    std::cout << tstep-1 << " -- " << t_eq/rv.dt << std::endl;
	return t_eq;
}
