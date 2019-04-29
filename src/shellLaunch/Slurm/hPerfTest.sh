#!/bin/bash

nxs=
tpbs=
gpuas=

affinity()
{
    export nxs="5e6 2e7 4e7 6e7"
    export tpbs="64 128 256 512 768 1024"
    gStep=20
    gEnd=$(( 10*$gStep ))
    export gpuas=$(seq 0 $gStep $gEnd) 
}

sweep() {
    ni=""
    a=0
    nb=$(printf "%.f" "1e6")
    for i in $(seq 10); do
        a=$(( $nb*(9+$i)+$a ))
        ni+="$a "
        echo $a
    done
    export nxs=$ni
    export tpbs=$(seq 128 128 1024)
    gStart=100
    gStep=5
    gEnd=$(( 10*$gStep+$gStart ))
    export gpuas=$(seq $gStart $gStep $gEnd)
}

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
nper=${6-:$SLURM_NNODES}
npr=$(( $nper*$nprocs ))

gStep=20
gEnd=$(( 10*$gStep ))
bindir=${SRCPATH}/bin
testdir=${SRCPATH}/oneD/tests

$1

mkdir -p $opath
mkdir -p $LOGPATH

if [[ $# -lt 3 ]]; then
    echo "NOT ENOUGH ARGUMENTS"
    return 0
fi

eq=$2
tf=$3
sc=$4
rnode=${5-:0}
hname=$(hostname)
hnm=${hname%%.*}

confile="${testdir}/${eq}Test.json"
execfile="${bindir}/${eq}"
logf="${LOGPATH}/${eq}_${sc}_AFF_${hnm}.log"
rm -f $logf
touch $logf

echo $tpbs $nxs $gpuas $logf 

for tpb in $tpbs
do
    for nx in $nxs
    do
        snx0=$SECONDS
        for g in $gpuas
        do
            echo -------- START ------------ | tee -a $logf
            lx=$(( $nx/10000 + 1 ))
            S0=$SECONDS

            echo "srun -N $SLURM_NNODES -n $npr -r $rnode $execfile $sc $confile $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf"
            srun -N $SLURM_NNODES -n $npr -r $rnode $execfile $sc $confile $opath tpb $tpb gpuA $g nX $nx lx $lx tf $tf 2>&1 | tee -a $logf

            echo --------------------------- | tee -a $logf
            echo -e "len|\t eq|\t sch|\t tpb|\t gpuA|\t nX" | tee -a $logf
            s1=$(( $SECONDS-$S0 ))
            echo -e "$lx|\t $eq|\t $sc|\t $tpb|\t $g|\t $nx\t took $s1" | tee -a $logf
            echo --------- END -------------- | tee -a $logf
        done
        snx1=$(( $SECONDS-$snx0 ))
        echo All together $snx1 secs | tee -a $logf
        snxout=$(( $snx1/60 ))
        echo $eq "|" $sc "|" $tpb "|" $nx :: $snxout >> $tfile 
    done
done
