#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

NO_OF_CORES=$(expr $SLURM_TASKS_PER_NODE \* $SLURM_JOB_NUM_NODES)

for i in `seq $NO_OF_CORES`; do
    srun -n 1 osprey worker alpha_angle.yaml -n 1 > osprey.$SLURM_JOB_ID.$i.log 2>&1 &
done
wait

