#!/bin/bash

#SBATCH -J shsweep						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p mime4								# name of partition or queue

#SBATCH -F ./nrg-nodes

#SBATCH -o shsweep.out					# name of output file for this submission script

#SBATCH -e shsweep.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

#SBATCH --time=7-00:00:00
​
#SBATCH --ntasks=40

# load any software environment module required for app
# run my jobs
/scratch/a1/mpich3i/bin/mpirun -bootstrap slurm --host compute-e-1,compute-e-2 -n 40   ./bin/euler S ./oneD/tests/eulerTest.json ./rslts tpb 64 gpuA 20 nX 200000 lx 21 tf 1

