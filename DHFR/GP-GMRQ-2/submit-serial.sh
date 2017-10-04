#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=4000

cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

source activate science
module load libGLU/9.0.0-foss-2016a-Mesa-11.2.1

osprey worker $config.yaml -n 150 --seed 1 > $config.$SLURM_JOBID.log 2>&1 


