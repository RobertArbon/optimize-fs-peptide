#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

osprey worker rmsd.yaml -n 11 --seed 42  > osprey.$SLURM_JOB_ID.log 2>&1 
