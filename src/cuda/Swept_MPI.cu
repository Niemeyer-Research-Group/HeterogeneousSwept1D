
// This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
#ifndef REAL
    #define REAL            float
    #define REALtwo         float2
    #define REALthree       float3
    #define SQUAREROOT(x)   sqrtf(x)

    #define ZERO            0.0f
    #define QUARTER         0.25f
    #define HALF            0.5f
    #define ONE             1.f
    #define TWO             2.f
#else

    #define ZERO            0.0
    #define QUARTER         0.25
    #define HALF            0.5
    #define ONE             1.0
    #define TWO             2.0
    #define SQUAREROOT(x)   sqrt(x)
#endif



/**
    Builds an upright triangle using the swept rule.

    Upright triangle using the swept rule.  This function is called first using the initial conditions or after results are read out using downTriange.  In the latter case, it takes the result of down triangle as IC.

    @param IC Array of initial condition values in order of spatial point.
    @param outRight Array to store the right sides of the triangles to be passed.
    @param outLeft Array to store the left sides of the triangles to be passed.
*/
__global__
void
upTriangle(const REALthree *IC, REALthree *outRight, REALthree *outLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tididx = threadIdx.x + 2; //Block Thread ID
    int tidxTop = tididx + dimens.base; //
    int k=4;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

    __syncthreads();

	if (threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
		temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	while (k < (blockDim.x>>1))
	{
		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, false, false);
		}

        k+=2;
		__syncthreads();

		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
		}

		k+=2;
		__syncthreads();

	}
    // Passes right and keeps left
    writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
    @param inLeft Array of left edges seeding solution vector.
*/
__global__
void
downTriangle(REALthree *IC, const REALthree *inRight, const REALthree *inLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    int k = dimens.hts[2];

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    __syncthreads();

	while(k>1)
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}

        k-=2;
        __syncthreads();

        if (!truth.x && !truth.w && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);

        }

		k-=2;
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}


/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param inRight Array of right edges seeding solution vector.
    @param inLeft Array of left edges seeding solution vector.
    @param outRight Array to store the right sides of the triangles to be passed.
    @param outLeft Array to store the left sides of the triangles to be passed.
    @param Full True if there is not a node run on the CPU, false otherwise.
*/
__global__
void
wholeDiamond(const REALthree *inRight, const REALthree *inLeft, REALthree *outRight, REALthree *outLeft, const bool split)
{

    extern __shared__ REALthree temper[];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;

    char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    if (split)
    {
        gid += blockDim.x;
        truth.x = false, truth.y = false, truth.z = false, truth.w = false;
    }

    readIn(temper, inRight, inLeft, threadIdx.x, gid);

    __syncthreads();

    int k = dimens.hts[0];

    if (tididx < (dimens.base-dimens.hts[2]) && tididx >= dimens.hts[2])
    {
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
    }

    __syncthreads();

    while(k>4)
    {
        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();
    }

    // -------------------TOP PART------------------------------------------

    if (!truth.w  &&  !truth.x)
    {
        temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
    }

    __syncthreads();

    if (tididx > 3 && tididx <(dimens.base-4))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
	}


    k=6;
	__syncthreads();

	while(k<dimens.hts[4])
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k+=2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}
		k+=2;
		__syncthreads();
	}

    if (split)
    {
        writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
    }
    else
    {
        writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
    }
}


//Split one is always first.
__global__
void
splitDiamond(REALthree *inRight, REALthree *inLeft, REALthree *outRight, REALthree *outLeft)
{
    extern __shared__ REALthree temper[];

    //Same as upTriangle
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    int k = dimens.hts[2];

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

    const char4 truth = {gid == dimens.hts[0], gid == dimens.hts[1], gid == dimens.hts[2], gid == dimens.hts[3]};

    __syncthreads();

    if (truth.z)
    {
        temper[tididx] = dbd[0];
        temper[tidxTop] = dbd[0];
    }
    if (truth.y)
    {
        temper[tididx] = dbd[1];
        temper[tidxTop] = dbd[1];
    }

    __syncthreads();

    while(k>0)
    {

        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();

        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();
    }

    if (!truth.y && !truth.z && threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
	}

	__syncthreads();
    k=4;

    //The initial conditions are timslice 0 so start k at 1.
    while(k<dimens.hts[2])
    {
        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);

        }

        k+=2;
        __syncthreads();

        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }
        k+=2;
        __syncthreads();

    }

	writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}


using namespace std;

__host__
__forceinline__
REAL
energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

// This  is part of the MPI routine now kinda.
__host__
void
CPU_diamond(REALthree *temper, int htcpu[5])
{

    omp_set_num_threads(8);

    temper[htcpu[2]] = bd[0];
    temper[htcpu[2]+dimz.base] = bd[0];

    temper[htcpu[1]] = bd[1];
    temper[htcpu[1]+dimz.base] = bd[1];

    //Splitting it is the whole point!
    for (int k = htcpu[0]; k>0; k-=4)
    {
        #pragma omp parallel for
        for(int n = k; n<(dimz.base-k); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
            }
        }

        #pragma omp parallel for
        for(int n = k-2; n<(dimz.base-(k-2)); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n] += eulerFinalStep(temper, n+dimz.base, n==htcpu[3],(n==htcpu[0]));
            }
        }
    }

    #pragma omp parallel for
    for(int n = 4; n < (dimz.base-4); n++)
    {
        if (n!=htcpu[1] && n!=htcpu[2])
        {
            temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
        }
    }

    //Top part.
    for (int k = 6; k<htcpu[2]; k+=4)
    {
        #pragma omp parallel
        for(int n = k; n<(dimz.base-k); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n] += eulerFinalStep(temper, n + dimz.base, (n==htcpu[3]), (n==htcpu[0]));
            }
        }

        #pragma omp parallel for
        for(int n = (k+2); n<(dimz.base-(k+2)); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
            }
        }
    }
}

//The wrapper that calls the routine functions.
double
sweptWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, const int cpu,
    REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    const size_t smem = (2*dimz.base)*sizeof(REALthree);
    const int cpuLoc = dv-tpb;

    int htcpu[5];
    for (int k=0; k<5; k++) htcpu[k] = dimz.hts[k]+2;

	REALthree *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

	cudaMalloc((void **)&d_IC, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REALthree)*dv);
    cudaMalloc((void **)&d2_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REALthree)*dv);

	cudaMemcpy(d_IC,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    double t_eq;
    double twrite = freq - QUARTER*dt;

	// Call the kernels until you reach the final time

    if (cpu)
    {
        cout << "Hybrid Swept scheme" << endl;

        REALthree *h_right, *h_left;
        REALthree *tmpr = (REALthree *) malloc(smem);
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REALthree), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REALthree), cudaHostAllocDefault);

        t_eq = t_fullstep;

        cudaStream_t st1, st2, st3;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);

        //Split Diamond Begin------

        wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

        CPU_diamond(tmpr, htcpu);

        for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
        cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

        // CPU Part End -----

        while(t_eq < t_end)
        {
            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            //Split Diamond Begin------

            wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

            cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
            cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

            cudaStreamSynchronize(st2);
            cudaStreamSynchronize(st3);

            // CPU Part Start -----

            for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

            CPU_diamond(tmpr, htcpu);

            for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

            cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

            // CPU Part End -----

            // Automatic synchronization with memcpy in default stream

            //Split Diamond End------

            t_eq += t_fullstep;

    	    if (t_eq > twrite)
    		{
    			downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

                fwr << "Density " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << "Velocity " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << "Energy " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
                fwr << endl;

                fwr << "Pressure " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
                fwr << endl;

                upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    			splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

                t_eq += t_fullstep;

                twrite += freq;
    		}
        }

        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        free(tmpr);

	}
    else
    {
        cout << "GPU only Swept scheme" << endl;
        splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {

            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
            //So it always ends on a left pass since the down triangle is a right pass.
            t_eq += t_fullstep;

            if (t_eq > twrite)
    		{
    			downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

                fwr << "Density " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << "Velocity " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << "Energy " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
                fwr << endl;

                fwr << "Pressure " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
                fwr << endl;

    			upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    			splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
        }
    }

    downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
    cudaFree(d2_right);
	cudaFree(d2_left);

    return t_eq;
}


