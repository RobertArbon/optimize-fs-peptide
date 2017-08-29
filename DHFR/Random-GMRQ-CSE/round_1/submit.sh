#!/bin/bash -login
#SBATCH --job-name=rand-cse
#SBATCH --time=12:00:00
#SBATCH --mem 20000
##SBATCH -N 1                      # number of nodes
##SBATCH -n 2                    # number of cores

#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1

#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERRo

cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
echo $SLURM_JOB_CPUS_PER_NODE

python run_from_trials.py 
 


