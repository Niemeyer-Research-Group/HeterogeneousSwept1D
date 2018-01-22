#!/bin/zsh

#$ -cwd
#$ -N hSweptSingleRun
#$ -q mime4
#$ -pe mpich2 12
#$ -j y
#$ -R y
#$ -l h=compute-e-1

hostname

gpua=5.0
tpbs=320

for eq in heat euler
do
	for sc in S C
    do
        $MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ../bin/$eq $sc ../tests/"$eq"Test.json ../rslts tpb $tpbs nX 504012 gpuA $gpua lx 60
        echo $eq $sc
    done
done
