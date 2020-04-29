#!/bin/bash

#SBATCH -J hrf						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p mime4								# name of partition or queue

#SBATCH -F ./nrg-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --time=7-00:00:00

#SBATCH -o hrf.out					# name of output file for this submission script

#SBATCH -e hrf.err					# name of error file for this submission script

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

echo Running initialization run...
/scratch/a1/mpich3i/bin/mpirun -n 40 --hostfile ./nrg-nodes ./bin/heat S ./oneD/tests/heatTest.json ./ tpb 64 gpuA 0 nX 10000 lx 10 tf 0.1
echo Finished initialization run...
rm ./tHeatS.csv

for eq in heat euler
do
	for sc in S C
	do
		tfile="./rslts/timecost-${eq}-${sc}"
		touch tfile
		echo len, eq, sch, tpb, gpuA, nX, tf, tcost >> $tfile
		if [ $eq == 'euler' ]
		then
			tf=0.04
		else
			tf=0.4
		fi
			for tpb in 64 96 128 192 256 384 512 768 1024
			do
				for g in $(seq 0 5 110)
				do
					for nx in  10000 50000 100000 500000 1000000 2000000 5000000 8000000 10000000
					do
							echo -------- START ------------
							lx=$(($nx / 10000))
							rtime="$( TIMEFORMAT='%R';time ( /scratch/a1/mpich3i/bin/mpirun -n 40 --hostfile ./nrg-nodes ./bin/$eq $sc ./oneD/tests/"$eq"Test.json $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf ) 2>&1 1>/dev/null )"
							echo len, eq, sch, tpb, gpuA, nX, tf \n
							echo $lx, $eq, $sc, $tpb, $g, $nx, $tf, $rtime
							echo $lx, $eq, $sc, $tpb, $g, $nx, $tf, $rtime >> $tfile
							echo -------- END ------------
					done
			done
		done
	done
done
