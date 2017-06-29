#!/bin/bash

N_CORES=5
job_n=1
for job in `ls *.yaml` 
do
     name=${job%.*}
     name=${name#config_random-}
     echo $name
     mpirun -np $N_CORES --output-filename $name osprey worker -n 40 $job   
done
