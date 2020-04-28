#!/bin/bash

#SBATCH -J walker						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p mime4								# name of partition or queue

#SBATCH -F ./nrg-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --time=7-00:00:00

#SBATCH -o walker.out					# name of output file for this submission script

#SBATCH -e walker.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

#/scratch/a1/mpich3i/bin/mpirun -bootstrap slurm --host compute-e-1,compute-e-2 -n 40   ./bin/euler S ./oneD/tests/eulerTest.json ./rslts tpb 64 gpuA 20 nX 200000 lx 21 tf 1

echo $SLURM_JOB_ID

hostname
ls rslts
rm rslts/* || true
ls rslts
opath="./rslts"
mkdir -p $opath

for sc in S C
do
	for eq in heat euler
	do
		if [ $eq == 'euler' ]
		then
			tf=0.06
		else
			tf=0.6
		fi
			for tpb in 64 96 128 192 256 384 512 768 1024
			do
				for g in $(seq 0 5 100)
				do
					for nx in  500000 1000000 5000000 10000000
					do
							echo -------- START ------------
							# nx=$($nxStart * 2**$x)
							lx=$(($nx / 10000))
							S0=$SECONDS
							/scratch/a1/mpich3i/bin/mpirun -n 40 --hostfile ./nrg-nodes ./bin/$eq $sc ./oneD/tests/"$eq"Test.json $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf
							echo len, eq, sch, tpb, gpuA, nX, tf \n
							s1=$(($SECONDS-$S0))
							echo $lx, $eq, $sc, $tpb, $g, $nx $tf took $s1
							echo -------- END ------------
							sleep 0.05
					done
			done
		done
	done
done
