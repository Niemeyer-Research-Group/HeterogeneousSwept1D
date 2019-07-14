#!/bin/zsh

#$ -cwd
#$ -N hAffine
#$ -q mime4
#$ -pe mpich3i 40
#$ -j y
#$ -R y
#$ -o ../../../.runOut/hAffine.out
#$ -l h='compute-e-[1-2]

export JID=$JOB_ID

echo $JOB_ID

hostname

ls rslts

rm rslts/* || true

ls rslts

tfile="../../rsltlog/otime.dat"
opath="../../rslts"
tpath=$(dirname tfile)

gEnd=$((2*$NSLOTS))
gStep=$(($gEnd/10))
bindir=../../bin
testdir=../../oneD/tests

rm $tfile || true

eqs=(heat euler)
tfs=(2.0 0.08)
nxs=(1e5 1e6 1e7)

mkdir -p $opath
mkdir -p $(dirname $tfile)
hname=$(hostname)

for ix in $(seq 2)
do
	eq=$eqs[$ix]
	tf=$tfs[$ix]
	for sc in S C
	do
        logf="${tpath}/${eq}_${sc}_AFF_${hname}.log"
        touch $logf
		for t in $(seq 1 12)
		do
			tpb=$((64*$t))
			snx0=$SECONDS
			for nx in $nxs
			do
				for g in $(seq 0 $gStep $gEnd) 
				do
					echo -------- START ------------
					lx=$(($nx/10000 + 1))
					S0=$SECONDS
					
					$MPIPATH/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines $bindir/$eq $sc $testdir/"$eq"Test.json $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf 2>&1 | tee -a $logf

					echo ---------------------------
					echo len, eq, sch, tpb, gpuA, nX
					s1=$(($SECONDS-$S0))
					echo $lx, $eq, $sc, $tpb, $g, $nx took $s1
					echo --------- END --------------
				done
				snx1=$(($SECONDS-$snx0))
                snxout=$(($snx1/60.0))
				echo All together $snx1 secs
                echo $eq "|" $sc "|" $tpb "|" $g :: $snxout >> $tfile
				done
			done
		done
	done
done
