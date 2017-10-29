# HeterogeneousSwept1D

<!-- TOC -->

- [HeterogeneousSwept1D](#heterogeneousswept1d)
    - [Dependencies:](#dependencies)
- [To Run](#to-run)

<!-- /TOC -->

Research in applying the swept rule for PDEs in one spatial dimension to a distributed cluster with CPUs and GPUs.

Python files in main folder are entry points.

## Dependencies:

[jsoncpp](https://github.com/open-source-parsers/jsoncpp) - jsoncpp.sh in main folder will download, amalgamate and put in utilities folder.

[palettable](http://www.palettable.io) - to run some of the plotting scripts

CUDA 7.5 or greater
MPI 2

# To Run

make with makefile in src folder

Run from src folder 
mpirun -np [nprocs] ./bin/[executable] [scheme (C or S) for classic or swept] [path to json with run specifications (eg. tests folder)] [path to output folder] [additional options as tokens eg. tpb 32 or lx 85]

python script stdExp will run standard experiment.

json files in result folder are coded [s or t for solution or timing[problem eg Euler or Heat][_rank]or[S or C for swept or classic]]