#!/bin/bash

#$ -cwd
#$ -N hSweptSingleRun
#$ -q mime4
#$ -pe mpich2 40
#$ -j y
#$ -R y
#$ -l h='compute-e-[1-2]

hname=`hostname` 
eq=$1
sch=$2
dims=$3
MPIPATH=/usr/local
NSLOTS=$4
ex=$5
HOSTH=`echo $hname | grep hpc`
if [[ -n $HOSTH ]]; then 
    machinef="-machinefile $TMPDIR/machines"
fi

$MPIPATH/bin/mpirun -np $NSLOTS $machinef ./bin/$eq $sch $dims/tests/"$eq"Test.json $dims/rslts $ex

