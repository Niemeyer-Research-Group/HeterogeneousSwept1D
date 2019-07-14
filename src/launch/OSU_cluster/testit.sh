#!/bin/bash

#$ -cwd
#$ -N hSweptSingleRun
#$ -q mime4
#$ -pe mpich2 40
#$ -j y
#$ -R y
#$ -l h='compute-e-[1-2]

hostname

eq=$1

sch=$2

ex=$3

echo $ex

$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ../bin/$eq $sch ../tests/"$eq"Test.json ../rslts $ex

