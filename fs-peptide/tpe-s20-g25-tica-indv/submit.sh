#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
#PBS -N tpe-indv 
#PBS -V


cd $PBS_O_WORKDIR
NO_OF_CORES=`cat $PBS_NODEFILE | egrep -v '^#'\|'^$' | wc -l | awk '{print $1}'`
source activate ml4dyn
i=1
for f in `ls *.yaml`; do
    if [ $i -le $NO_OF_CORES ]
    then
    osprey worker $f -n 200 --seed $i > osprey.$PBS_JOBID.$i-$f.log 2>&1 &
    fi
    (( i += 1 ))
done


#i=1
#for f in `ls *.yaml`
#do
#  if [ $i -le $NO_OF_CORES ]
#  then
#     echo $f, $i
#  fi
#  (( i += 1 ))
#done
#~
