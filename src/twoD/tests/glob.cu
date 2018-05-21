#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_functions.h>

__device__ int *boyo;

__global__ void gimme()
{
    printf("%d \n", boyo[threadIdx.x]);
}

int main(int argc, char *argv[])
{
    int mb[32];
    cudaMalloc((void **) &boyo, 32*4);
    for (int i=0; i<32; i++)
    {
        mb[i] = i*i;
    }
    cudaMemcpy(boyo, &mb, 32*4, cudaMemcpyHostToDevice);
    gimme <<<1, 32>>> ();

    cudaFree(boyo);
    return 0;

}