


//Read in the data from the global right/left variables to the shared temper variable.
__device__ __forceinline__
void readIn(double *temp, const double *rights, const double *lefts, int td, int gd)
{
	int leftidx = disc.ht + (((td>>2) & 1) * disc.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = disc.ht + (((td>>2) & 1) * disc.base) + ((td>>2)<<1) + (td & 3);

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

__device__ __forceinline__
void writeOutRight(double *temp, double *rights, double *lefts, int td, int gd, int bd)
{
	int gdskew = (gd + bd) & disc.idxend;
    int leftidx = (((td>>2) & 1)  * disc.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (disc.base - 6) + (((td>>2) & 1)  * disc.base) + (td & 3) - ((td>>2)<<1);
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}

__device__ __forceinline__
void writeOutLeft(double *temp, double *rights, double *lefts, int td, int gd, int bd)
{
	int gdskew = (gd - bd) & disc.idxend;
    int leftidx = (((td>>2) & 1)  * disc.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (disc.base-6) + (((td>>2) & 1)  * disc.base) + (td & 3) - ((td>>2)<<1);
	rights[gd] = temp[rightidx];
	lefts[gdskew] = temp[leftidx];
}

__device__ __forceinline__
double stutterStep(const double *u, const int loc[5])
{
	return u[loc[2]] - disc.dt_half * (convect(u[loc[1]], u[loc[3]]) + secondDer(u[loc[1]], u[loc[3]], u[loc[2]]) +
		fourthDer(u[loc[0]], u[loc[1]], u[loc[2]], u[loc[3]], u[loc[4]]));
}

__device__
double finalStep(const double *u, const int loc[5])
{
	return (-disc.dt * (convect(u[loc[1]], u[loc[3]]) + secondDer(u[loc[1]], u[loc[3]], u[loc[2]]) +
		fourthDer(u[loc[0]], u[loc[1]], u[loc[2]], u[loc[3]], u[loc[4]])));
}

//Classic
template <bool FINAL>
__global__ void
classicKS(const double *ks_in, double *ks_out)
{
	auto gidz = idxes<5>();
	
	if (FINAL)	 ks_out[gidz.a[2]] +=  finalStep(ks_in, gidz.a);
	else 		 ks_out[gidz.a[2]]  =  stutterStep(ks_in, gidz.a);
}

__global__
void
upTriangle(const double *IC, double *outRight, double *outLeft)
{
	extern __shared__ double temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tididx = threadIdx.x + 2;

	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + disc.base;
		tid_bottom[k+2] = tididx + k;
	}

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

	__syncthreads();

	if (threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
		temper[tid_top[2]] = stutterStep(temper, tid_bottom);
	}

	__syncthreads();

	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
			temper[tididx] += finalStep(temper, tid_top);
		}

		step2 = k + 2;
		__syncthreads();

		if (threadIdx.x < (blockDim.x-step2) && threadIdx.x >= step2)
		{
			temper[tid_top[2]] = stutterStep(temper, tid_bottom);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

	writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);

}

__global__
void
downTriangle(double *IC, const double *inRight, const double *inLeft)
{
	extern __shared__ double temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x+2;

	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + disc.base;
		tid_bottom[k+2] = tididx + k;
	}

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

	__syncthreads();

	for (int k = (disc.ht-2); k>0; k-=4)
	{
		if (tididx < (disc.base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper, tid_bottom);
		}

		step2 = k-2;
		__syncthreads();

		if (tididx < (disc.base-step2) && tididx >= step2)
		{
			temper[tididx] += finalStep(temper, tid_top);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}

template <bool SPLIT>
__global__ void
wholeDiamond(double *inRight, double *inLeft, double *outRight, double *outLeft)
{
	extern __shared__ double temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 2;

	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + disc.base;
		tid_bottom[k+2] = tididx + k;
	}

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

	__syncthreads();

	for (int k = (disc.ht-2); k>0; k-=4)
	{
		if (tididx < (disc.base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper, tid_bottom);
		}

		step2 = k-2;
		__syncthreads();

		if (tididx < (disc.base-step2) && tididx >= step2)
		{
			temper[tididx] += finalStep(temper, tid_top);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    //-------------------TOP PART------------------------------------------

	if (threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
		temper[tid_top[2]] = stutterStep(temper, tid_bottom);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
			temper[tididx] += finalStep(temper, tid_top);
		}

		step2 = k+2;
		__syncthreads();

		if (threadIdx.x < (blockDim.x-step2) && threadIdx.x >= step2)
		{
			temper[tid_top[2]] = stutterStep(temper, tid_bottom);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	if (SPLIT)
	{
		writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
	}
	else
	{
		writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
	}
}

double
classicWrapper(double *aState)
{
	std::cout << " -- Classic Array -- " << std::endl;

	const int aSize = sizeof(double) * rv.dv;
	int tstep = 0;
	// I'm just going to malloc in here.

    double *dks_in, *dks_out;

    cudaMalloc((void **)&dks_in,  aSize);
    cudaMalloc((void **)&dks_out, aSize);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dks_in, aState, aSize, cudaMemcpyHostToDevice);
    double t_eq = 0.0;

    while (t_eq <= rv.tf)
    {
        classicKS <false> <<< rv.bks, rv.tpb >>> (dks_in, dks_out);
        classicKS <true>  <<< rv.bks, rv.tpb >>> (dks_out, dks_in);
		tstep += 4;
        t_eq += rv.dt;
		
    }

    cudaMemcpy(aState, dks_in, aSize, cudaMemcpyDeviceToHost);
	std::cout << tstep << std::endl;

    cudaFree(dks_in);
    cudaFree(dks_out);

    return t_eq;
}


//The host routine.
double
sweptWrapper(double *aState)
{
	std::cout << " -- Swept Array -- " << std::endl;
	
	const size_t szSt = sizeof(double); 
    const size_t smem = (2*rv.tpb+8)*szSt;
    
    std::cout << "SHARED MEMORY: " << smem << std::endl;

	double *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;
	const int aSize = sizeof(double) * rv.dv;

	cudaMalloc((void **)&d_IC, 		aSize);
	cudaMalloc((void **)&d0_right, 	aSize);
	cudaMalloc((void **)&d0_left, 	aSize);
	cudaMalloc((void **)&d2_right, 	aSize);
	cudaMalloc((void **)&d2_left, 	aSize);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC, aState, aSize, cudaMemcpyHostToDevice);
	//Start the counter and start the clock.

	//Every other step is a full timestep and each cycle is half tpb steps.
	double t_fullstep = 0.25 * rv.dt * (double)rv.tpb;
	double t_eq;

	t_eq = t_fullstep;

	upTriangle <<< rv.bks, rv.tpb, smem >>> (d_IC, d0_right, d0_left);

	//Split
	wholeDiamond <true> <<< rv.bks, rv.tpb, smem >>> (d0_right, d0_left, d2_right, d2_left);

	// Call the kernels until you reach the iteration limit.
	while(t_eq < rv.tf)
	{
		wholeDiamond <false> <<< rv.bks, rv.tpb, smem >>> (d2_right, d2_left, d0_right, d0_left);

		//Split
		wholeDiamond <true> <<< rv.bks, rv.tpb, smem >>> (d0_right, d0_left, d2_right, d2_left);

		t_eq += t_fullstep;
	}

	downTriangle <<< rv.bks, rv.tpb, smem >>> (d_IC, d2_right, d2_left);

	cudaMemcpy(aState, d_IC, aSize, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
	cudaFree(d2_right);
	cudaFree(d2_left);
	std::cout << t_eq/rv.dt << std::endl;

	return t_eq;
}

