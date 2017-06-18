#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -N osprey_test
#PBS -V


cd $PBS_O_WORKDIR
NO_OF_CORES=`cat $PBS_NODEFILE | egrep -v '^#'\|'^$' | wc -l | awk '{print $1}'`
source activate ml4dyn
for i in `seq $NO_OF_CORES`; do
    osprey worker config_random.yaml -n 100 --seeds $i > osprey.$PBS_JOBID.$i.log 2>&1 &
done
wait
