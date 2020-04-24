#!/bin/bash

#SBATCH -J testjob						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p mime4								# name of partition or queue

#SBATCH -F ./test-nrg-nodes

#SBATCH -o testjob.out					# name of output file for this submission script

#SBATCH -e testjob.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

#SBATCH --time=7-00:00:00
​
#SBATCH --ntasks=40

# load any software environment module required for app
# run my jobs
for i in 10000 20000
do
/scratch/a1/mpich3i/bin/mpirun -bootstrap slurm --host compute-e-3,compute-e-4 -n 40   ./bin/euler S ./oneD/tests/eulerTest.json ./rslts tpb 64 gpuA 0 nX $i lx 21 tf 0.5
done
