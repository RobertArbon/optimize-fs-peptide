#!/bin/bash
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00
#PBS -m a

source activate ml4dyn
cd $PBS_O_WORKDIR
osprey worker $input -n 1 --seed 1 > $input.log 2>&1 

