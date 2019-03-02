#!/bin/bash

## SLURM NOT GRID
RPATH=$(python3 -c "import os; print(os.path.realpath('$0'))")
THISPATH=$(dirname $RPATH)
cd $THISPATH
SHPATH=$(dirname $THISPATH)
SRCPATH=$(dirname $SHPATH)
SWEEPPATH=$(dirname $SRCPATH)
LOGPATH="${SRCPATH}/rsltlog"
tfile="${LOGPATH}/otime.dat"
opath="${SRCPATH}/rslts"
nprocs=$(( $(nproc)/2 ))
npr=$(( $SLURM_NNODES*$nprocs ))

gStep=20
gEnd=$(( 10*$gStep ))
bindir=${SRCPATH}/bin
testdir=${SRCPATH}/oneD/tests

nxs="5e6 2e7 4e7 6e7"
tpbs="64 128 256 512 768 1024"

mkdir -p $opath
mkdir -p $LOGPATH

if [[ $# -lt 3 ]]; then
    echo "NOT ENOUGH ARGUMENTS"
    return 0
fi

eq=$1
tf=$2
sc=$3
hname=$(hostname)

confile="${testdir}/${eq}Test.json"
execfile="${bindir}/${eq}"
logf="${LOGPATH}/${eq}_${sc}_AFF_${hname}.log"
touch $logf

for tpb in $tpbs
do
    for nxi in $nxs
    do
        snx0=$SECONDS
        printf -v nx "%.f" "$nxi"
        for g in $(seq 0 $gStep $gEnd) 
        do
            echo -------- START ------------ | tee -a $logf
            lx=$(( $nx/10000 + 1 ))
            S0=$SECONDS
            
            srun -N $SLURM_NNODES -n $npr $execfile $sc $confile $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf 2>&1 | tee -a $logf

            echo --------------------------- | tee -a $logf
            echo -e "len|\t eq|\t sch|\t tpb|\t gpuA|\t nX" | tee -a $logf
            s1=$(( $SECONDS-$S0 ))
            echo -e "$lx|\t $eq|\t $sc|\t $tpb|\t $g|\t $nx\t took $s1" | tee -a $logf
            echo --------- END -------------- | tee -a $logf
        done
        snx1=$(( $SECONDS-$snx0 ))
        echo All together $snx1 secs | tee -a $logf
        snxout=$(( $snx1/60 ))
        echo $eq "|" $sc "|" $tpb "|" $nxi :: $snxout >> $tfile 
    done
done
