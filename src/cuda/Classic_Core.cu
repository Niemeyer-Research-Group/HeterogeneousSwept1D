/**
    Classic kernel for simple decomposition of spatial domain.

    Uses dependent variable values in euler_in to calculate euler out.  If it's the predictor step, finalstep is false.  If it is the final step the result is added to the previous euler_out value because this is RK2.

    @param euler_in The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization.
    @param euler_out The working array from the kernel call before last which either stores the predictor values or the full step values after the RHS is added into the solution.
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__
void
classicEuler(REALthree *euler_in, REALthree *euler_out, const bool finalstep)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    if (truth.x)
    {
        euler_out[gid] = dbd[0];
        return;
    }
    else if (truth.w)
    {
        euler_out[gid] = dbd[1];
        return;
    }

    if (finalstep)
    {
        euler_out[gid] += eulerFinalStep(euler_in, gid, truth.y, truth.z);
    }
    else
    {
        euler_out[gid] = eulerStutterStep(euler_in, gid, truth.y, truth.z);
    }
}


//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end,
    REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    REALthree *dEuler_in, *dEuler_out;

    cudaMalloc((void **)&dEuler_in, sizeof(REALthree)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALthree)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    cout << "Classic scheme" << endl;

    double t_eq = 0.0;
    double twrite = freq - QUARTER*dt;

    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

            fwr << "Density " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
            fwr << endl;

            fwr << "Velocity " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].y/T_f[k].x << " ";
            fwr << endl;

            fwr << "Energy " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
            fwr << endl;

            fwr << "Pressure " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
            fwr << endl;

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dEuler_in);
    cudaFree(dEuler_out);

    return t_eq;

}