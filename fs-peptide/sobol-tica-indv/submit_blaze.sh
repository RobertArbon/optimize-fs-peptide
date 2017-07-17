#!/bin/bash

N_CORES=6
job_n=1
for job in `ls *.yaml` 
do
     if [ $job_n -le 5 ] 
     then 
         name=${job%.*}
         name=${name#config-}
         echo $name
         #mpirun -np $N_CORES --output-filename $name osprey worker -n 33 $job   
         osprey worker -n 200 $job &> $name.log &
     fi 
     (( job_n += 1 ))
done
