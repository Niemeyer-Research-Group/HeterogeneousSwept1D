#!/bin/zsh

#$ -cwd
#$ -N hSweptTest
#$ -q mime4
#$ -pe mpich3i 20
#$ -j y
#$ -R y
#$ -o ../.runOut/hSweptTest.out
#$ -l h=compute-e-1

export JID=$JOB_ID

echo $JOB_ID

hostname

ls rslts

rm rslts/*

ls rslts

tfile="trslt/otime.dat"

rm $tfile

for eq in euler heat
do
	for sc in S C
	do
		for t in $(seq 6 9)
		do
			for dvt in $(seq 0 1)
			do
				tpbz=$((2**$t))
				tpb=$(($tpbz + 0.5*$dvt*$tpbz))
				for g in $(seq 0 2 20)
				do
					snx0=$SECONDS
					for x in $(seq 16 23)
					do
						for dvx in $(seq 0 1)
						do
							echo -------- START ------------
							nxo=$((2**$x))
							nx=$(($nxo + 0.5*$dvx*(2**$x)))
							lx=$(($nxo/10000 + 1))
							S0=$SECONDS
							$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq $sc ./tests/"$eq"Test.json ./rslts tpb $tpb gpuA $g nX $nx lx $lx
							echo len, eq, sch, tpb, gpuA, nX
							s1=$(($SECONDS-$S0))
							echo $lx, $eq, $sc, $tpb, $g, $nx took $s1
							echo -------- END ------------
							sleep 0.05
						done
					done
					snx1=$(($SECONDS-$snx0))
					snxout=$(($snx1/60.0))
					echo $eq "|" $sc "|" $tpb "|" $g :: $snxout >> $tfile
				done
			done
		done
	done
done

