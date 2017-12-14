#!/bin/zsh

#$ -cwd
#$ -N hSweptSingleRun
#$ -q mime4
#$ -pe mpich2 12
#$ -j y
#$ -R y
#$ -l h=compute-e-1

hostname

eq=$1

$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq C ./tests/"$eq"Test.json ./rslts


