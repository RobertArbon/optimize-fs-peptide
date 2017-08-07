#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=rand-cse
#SBATCH --time=12:00:00
#SBATCH --mem=20000

cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST

python run_from_trials.py > $SLURM_JOB_ID.log
 


