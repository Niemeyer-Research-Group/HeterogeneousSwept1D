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
tfile="./trslt/otime.dat"
opath="./rslts"
rm $tfile || true

mkdir -p $opath
mkdir -p $(dirname $tfile)

tfs=(0.06 1.2)
nxStart=100000

for eq in euler heat
do
	if [ $eq == 'euler' ]
	then
		tf=0.06
	else
		tf=1.2
	fi
	for sc in S C
	do
		for t in $(seq 6 10)
		do
			for dvt in $(seq 0 1)
			do
				tpbz=$((2**$t))
				tpb=$(($tpbz + $dvt*$tpbz/2))
				for g in $(seq 0 5 100)
				do
					snx0=$SECONDS
					for x in $(seq 0 7)
					do
						if [[ $x -gt 0 && $x -lt 7 ]]
						then
							midpt=1
						else
							midpt=0
						fi
						for dvx in $(seq 0 $midpt)
						do
							echo -------- START ------------
							nxo=$(($nxStart * 2**$x))
							nx=$(($nxo + $dvx*$nxo/2))
							lx=$(($nxo/10000 + 1))
							S0=$SECONDS
							/scratch/a1/mpich3i/bin/mpirun -n 40 --hostfile ./nrg-nodes ./bin/$eq $sc ./oneD/tests/"$eq"Test.json $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf
							echo len, eq, sch, tpb, gpuA, nX, tf
							s1=$(($SECONDS-$S0))
							echo $lx, $eq, $sc, $tpb, $g, $nx $tf took $s1
							echo -------- END ------------
							sleep 0.05
						done
					done
					snx1=$(($SECONDS-$snx0))
					snxout=$(($snx1/60))
					echo "----------------------WRITING TIME-------------------------"
					echo $snxout
					echo $eq "|" $sc "|" $tpb "|" $g :: $snxout >> $tfile
					echo "----------------------END WRITING TIME----------------------"
				done
			done
		done
	done
done
