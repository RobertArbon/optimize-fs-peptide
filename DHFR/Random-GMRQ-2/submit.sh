#!/bin/bash
#SBATCH --array=1-20
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

osprey worker $config.yaml -n 10 --seed $SLURM_ARRAY_TASK_ID > $config.$SLURM_ARRAY_TASK_ID.log 2>&1 


