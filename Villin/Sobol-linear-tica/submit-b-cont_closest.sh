#!/bin/bash
#PBS -j oe
#PBS -l nodes=1:ppn=8
#PBS -l walltime=30:00:00
#PBS -m a

n_trials=13
cd $PBS_O_WORKDIR
NO_OF_CORES=`cat $PBS_NODEFILE | egrep -v '^#'\|'^$' | wc -l | awk '{print $1}'`
source activate ml4dyn
f=b-cont_closest.yaml
for i in `seq $NO_OF_CORES`; do
    osprey worker $f -n $n_trials --seed $i > $f-$i-$PBS_JOBID.log 2>&1 &
done
wait

