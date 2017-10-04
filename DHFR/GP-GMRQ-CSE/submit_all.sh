#!/bin/bash

for f in `ls *.yaml`
do
   name=${f%.yaml}
   echo $name
   #sbatch --export=config=$name --job-name=$name-cse submit.sh
   sbatch --export=config=$name --job-name=$name-cse submit-serial.sh
done
