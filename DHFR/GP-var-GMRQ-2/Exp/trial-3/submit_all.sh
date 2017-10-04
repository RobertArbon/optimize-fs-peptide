#!/bin/bash

for f in `ls *.yaml`
do
   name=${f%.yaml}
   if [ $name != 'alpha_angle' ]
   then
       echo $name
       sbatch --export=config=$name --job-name=$name submit-serial.sh
   fi
done
