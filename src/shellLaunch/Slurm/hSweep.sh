#!/bin/bash

## SLURM NOT GRID
RPATH=$(python3 -c "import os; print(os.path.realpath('$0'))")
THISPATH=$(dirname $RPATH)
cd $THISPATH
SHPATH=$(dirname $THISPATH)
SRCPATH=$(dirname $SHPATH)
SWEEPPATH=$(dirname $SRCPATH)

source $SHPATH/shfuncs

setvar "$@"

tfile="${SRCPATH}/trslt/otime.dat"
opath="${SRCPATH}/rslts"
nprocs=$(( $(nproc)/2 ))
npr=$(( $SLURM_NNODES*$nprocs ))
rm -f $tfile 
rm -rf $opath

for ix in $(seq 2)
do
	eq=$eqs[$ix]
	tf=$tfs[$ix]
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
							nx=$(($nxo + 0.5*$dvx*$nxo))
							lx=$(($nxo/10000 + 1))
							S0=$SECONDS
							$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines ../bin/$eq $sc ../tests/"$eq"Test.json ../rslts tpb $tpb gpuA $g nX $nx lx $lx tf $tf
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
