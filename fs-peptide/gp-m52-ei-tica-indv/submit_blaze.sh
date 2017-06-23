#!/bin/bash

N_CORES=6
job_n=1
for job in `ls *.yaml` 
do
     if [ $job_n -le 5 ] 
     then 
         name=${job%.*}
         name=${name#config_random-}
         echo $name
         mpirun -np $N_CORES --output-filename $name osprey worker -n 33 $job   
     fi 
     (( job_n += 1 ))
done
