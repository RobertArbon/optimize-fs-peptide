#!/bin/bash

for f in `ls *.yaml`
do
   name=${f%.yaml}
   echo $name
   #sbatch --export=config=$name --job-name=$name submit.sh
   sbatch --export=config=$name --job-name=$name submit-serial.sh
done
