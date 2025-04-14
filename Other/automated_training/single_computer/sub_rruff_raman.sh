#!/bin/bash
#SBATCH --nodes=1                  # Request 2 nodes
#SBATCH -t 100:00:00                # Request 12 hours in walltime.
#SBATCH -J 20250406r           # Job name
#SBATCH -o log_output              # output file
#SBATCH -e log_error               # error file
#SBATCH --ntasks=1                # total number of cores required. Nodes*24
#SBATCH --export=ALL

logname="log_"${PWD##*/}".txt"

train_rruff_raman.sh $SLURM_JOB_NAME

