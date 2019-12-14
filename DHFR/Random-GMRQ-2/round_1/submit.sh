#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

NO_OF_CORES=$(expr $SLURM_TASKS_PER_NODE \* $SLURM_JOB_NUM_NODES)

for i in `seq $NO_OF_CORES`; do
    osprey worker $config.yaml -n 20 --seed $i > osprey.$SLURM_JOB_ID.$i.log 2>&1 &
done
wait

