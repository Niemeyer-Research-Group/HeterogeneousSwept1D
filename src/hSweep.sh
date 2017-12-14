#!/bin/zsh

#$ -cwd
#$ -N hSweptTest
#$ -q mime4
#$ -pe mpich2 12
#$ -j y
#$ -R y
#$ -l h=compute-e-1

hostname

ls rslts

rm rslts/*

ls rslts

for eq in heat euler
do
	for sc in S C

		for t in $(seq 1 12)
		do
			tpb=$((64*$t))
			for g in $(seq 0 12)
			do
				for x in $(seq 15 24)
				do
					nx=$((2**$x))
					lx=$(($nx/10000 + 1))

					$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq $sc ./tests/"$eq"Test.json ./rslts tpb $tpb gpuA $g nX $nx lx $lx
				done
			done
		done
	done
done

