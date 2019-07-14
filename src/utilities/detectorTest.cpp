#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "gpuDetector.h"

int main(int argc, char *argv[]) {

    int rank, nprocs;
    bool gi;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int pos = 1;
    if (argc > 1) pos=atoi(argv[1]);

    gi = detector(rank, nprocs, pos);

    if (gi) printf("I have a GPU - Rank %d\n", rank);

    return 0;
}
