<<<<<<< HEAD
#!/bin/bash 
#PBS -j oe 
#PBS -l nodes=1:ppn=5,walltime=48:00:00
#! Mail to user if job aborts
#PBS -m a

###############################################################
### You should not have to change anything below this line ####
###############################################################

#! change the working directory (default is home directory)

cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID

source activate ml4dyn
python rescore_trials.py $nts > test.log 2>&1 
#python refit_trials.py > all_ts.log 2>&1 


=======
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

source activate science

python rescore_trials.py

>>>>>>> 8297abcaf3a2da36eb9a4de9525638a6207eb8ae
