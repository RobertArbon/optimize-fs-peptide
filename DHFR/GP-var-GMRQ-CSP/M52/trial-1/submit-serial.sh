#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5

cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

source activate science

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load libGLU/9.0.0-foss-2016a-Mesa-11.2.1

osprey worker $config.yaml -n 200 -j 5 --seed 1 > $config.$SLURM_JOBID.log 2>&1 


