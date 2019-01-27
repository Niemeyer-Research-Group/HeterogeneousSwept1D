#!/bin/zsh

#$ -cwd
#$ -N hSweptTest
#$ -q mime4
#$ -pe mpich3i 40
#$ -j y
#$ -R y
#$ -o ../.runOut/hSweptTest.out -e ../.runOut/hSweepErr.log
#$ -l h='compute-e-[1-2]

## Need to fix the strange duplicate of runs.

export JID=$JOB_ID

echo $JOB_ID

hostname

PREFIX=./oneD

ls $PREFIX/rslts

rm $PREFIX/rslts/* || true

ls $PREFIX/rslts

tfile=$PREFIX/trslt/otime.dat

rm $tfile || true

MPIEX=$MPIPATH/bin/mpirun

eqs=(euler heat)
tfs=(0.06 1.2)
nxStart=(5 7.5 10 25 50 75 100)

for ix in $(seq 2)
do
	eq=$eqs[$ix]
	tf=$tfs[$ix]
	for sc in S
	do
		for t in $(seq 6 9)
		do
			for dvt in $(seq 0 1)
			do
				tpbz=$((2**$t))
				tpb=$(($tpbz + 0.5*$dvt*$tpbz))
				for g in $(seq 20 4 60)
				do
					snx0=$SECONDS
					for x in $nxStart
					do
						echo -------- START ------------
						echo $x -- $g -- $eq -- $tf
						nx=$(($x * 100000))
						lx=$(($nx/10000 + 1))
						S0=$SECONDS
						$MPIEX -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq $sc $PREFIX/tests/"$eq"Test.json $PREFIX/rslts tpb $tpb gpuA $g nX $nx lx $lx tf $tf
						echo len, eq, sch, tpb, gpuA, nX
						s1=$(($SECONDS-$S0))
						echo $lx, $eq, $sc, $tpb, $g, $nx took $s1
						echo -------- END ------------
						sleep 0.5
					done
					snx1=$(($SECONDS-$snx0))
					snxout=$(($snx1/60.0))
					echo $eq "|" $sc "|" $tpb "|" $g :: $snxout >> $tfile
				done
			done
		done
	done
done
