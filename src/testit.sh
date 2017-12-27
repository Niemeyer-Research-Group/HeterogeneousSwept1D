#!/bin/zsh

#$ -cwd
#$ -N hSweptSingleRun
#$ -q mime4
#$ -pe mpich2 20
#$ -j y
#$ -R y
#$ -l h=compute-e-1

hostname

eq=$1

sch=$2

$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ./bin/$eq $sch ./tests/"$eq"Test.json ./rslts tpb 512  nX 867432 gpuA 5.0 lx 79.2523156451219993

