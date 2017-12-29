#!/bin/zsh

#$ -cwd
#$ -N hSweptTest
#$ -q mime4
#$ -pe mpich2 20
#$ -j y
#$ -R y
#$ -l h=compute-e-1

hostname

ls rslts

rm rslts/*

ls rslts

for eq in euler heat
do
	for sc in S C
	do
		for t in $(seq 1 12)
		do
			tpb=$((64*$t))
			for g in $(seq 0 2 12)
			do
				for x in $(seq 16 23)
				do
					for d in $(seq 0 1)
					do
						echo -------- START ------------
						nxo=$((2**$x))
						nx=$(($nxo + 0.5*$d*(2**$x)))
						lx=$(($nxo/10000 + 1))
						SECONDS=0
						$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq $sc ./tests/"$eq"Test.json ./rslts tpb $tpb gpuA $g nX $nx lx $lx
						echo len, eq, sch, tpb, gpuA, nX
						echo $lx, $eq, $sc, $tpb, $g, $nx took $SECONDS s
						echo -------- END ------------
						sleep 2
					done
				done
			done
		done
	done
done

