#!/bin/bash
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH -t 100:00:00                # Request 12 hours in walltime.
#SBATCH -J 20250413r           # Job name
#SBATCH -o log_%x.o%j              # output file
#SBATCH -e log_%x.e%j              # error file
#SBATCH --ntasks=1                # total number of cores required. Nodes*24
#SBATCH --export=ALL

command="srun train_rruff_raman.sh $SLURM_JOB_NAME "

echo $command

$command
