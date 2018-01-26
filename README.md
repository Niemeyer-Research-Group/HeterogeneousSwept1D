

# hSweep

[The swept rule](https://www.sciencedirect.com/science/article/pii/S0021999117309221?via%3Dihub) ([free](https://arxiv.org/abs/1705.03162)) is a communication avoiding algorithm for accelerating explicit solutions of time-dependent PDEs in parallel.
**hSweep** is a CUDA/MPI program that applies the swept rule to a distributed memory system with heterogeneous (CPU and GPU) architecutre.

Python files in src/\*D and shell scripts in src/shellLaunch are entry points, the program can also be run from the command line with mpirun.

## Build

Complete conf file in src.  This file contains the architecture specific arguments to the makefile.

Run make in src folder.  Make takes one optional option "DEF=-DNOS" if this option is set, the program does write out the equation solutions.

Run from src folder 
mpirun -np [nprocs] ./bin/[executable] [scheme (C or S) for classic or swept] [path to json with run specifications (eg. tests folder)] [path to output folder] [additional options as tokens eg. tpb 32 or lx 85]

python script runTiming will run standard performance experiment, runResult will run a single instance and plot the solution to the equation.

All results should be placed in the src/\*D/rslts folder by default since that is where the python scripts that analyze and plot them will look.
The equation solutions are written out to json files as dictionaries.  The file names are coded [s][equation name]][_rank].
The performance results are written to csv files and coded [t][equation name][S or C (for swept or classic)].

## Run

mpirun -np [NumProcs] [Executable] [Algorithm] [Config File] [Output Path] [Additional Args]

Algorithm: S for Swept, C for Classic

ConfigFile: [equation name][Description].json (example eulerTest.json found in src/\*D/tests

Output Path: Path should point to .dat folder in Results file. Name should include problem and precision. Format: Line 1 is length of full spatial domain, number of spatial points and grid step, Other rows are results with format: variable, time, value at each spatial point

Additional Args: These are the several standard arguments that every run must include on the command line or in the json configuration file.

| Argument  |  Meaning |
| --------- | -------- |
| nX | Number of points in the spatial domain (must be a power of 2 greater than 32)
| tpb | Number of points per node = threads per block (must be power of 2 greater than 32 less than #Spatial points)
| dt | timestep.
| tf | Ending time for the simulation.  NOTE: the swept rule will not stop at exactly this time but very near it depending on tpb.
| freq | Checkpoint at which program writes out simulation results (i.e. every 400s).  Example: if the finish time is 1000s and the output frequency is 400s, the program will write out the initial condition and the solution at two intermediate times (~400 and 800s)

## Evaluate

Python performance and accuracy analysis.

## Extend

Write your own equations and run them.

1.) Copy the newEquation.htpl

2.) The header file is just a skeleton, you'll need to fill in the details of the scheme in the functions, variables, and #defines provided, and you'll likely need to define new functions.

3.) Then run make and the executable.

## Dependencies:

[gitpython](http://gitpython.readthedocs.io/en/stable/intro.html)

CUDA 7.5 or greater
MPI 2 or greater

## Additional Details
### Important

gpuDetect 
why doesn't it run the number of spatial pts I described?

### Extra

Maybe in the wiki.

## ToDo
- Kill "REAL" just do double
- Update conf and makefile.
- Compart to Extend. protoype 
- Complete this README, push and version.
