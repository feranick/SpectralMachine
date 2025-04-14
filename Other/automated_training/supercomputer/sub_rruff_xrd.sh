#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 18:00:00
#SBATCH -J 20250413x
#SBATCH -o %x.o%j
#SBATCH -A m526
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -q regular
#SBATCH --mail-user=ferralis@mit.edu
#SBATCH --mail-type=ALL

exe=/global/homes/f/feranick/ml/train_rruff_xrd.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

command="srun -n 4 --cpu-bind=cores --gpu-bind=none train_rruff_xrd.sh $SLURM_JOB_NAME"

echo $command

$command
