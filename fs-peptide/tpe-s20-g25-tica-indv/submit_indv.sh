#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -V

cd $PBS_O_WORKDIR

source activate ml4dyn

name=${INPUT%.*}
name=${name#config-}
osprey worker $INPUT -n 200 --seed $SEEDNUM > osprey.$PBS_JOBID.$name.log 2>&1 &


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
