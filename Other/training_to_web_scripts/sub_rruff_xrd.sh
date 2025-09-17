#!/bin/bash
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH -t 100:00:00                # Request 12 hours in walltime.
#SBATCH -J SK_Powder           # Job name
#SBATCH -o log_%x.o%j              # output file
#SBATCH -e log_%x.e%j              # error file
#SBATCH --ntasks=1                # total number of cores required. Nodes*24
#SBATCH --export=ALL

#command="srun train_rruff_raman.sh $SLURM_JOB_NAME "
#command="srun train_rruff_xrd.sh $1"

command="srun ../train_rruff_xrd.sh $(basename $(pwd))"

echo $command

$command
