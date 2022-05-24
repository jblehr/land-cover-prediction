#!/bin/bash

#SBATCH --job-name=job_start
#SBATCH --ntasks=1
#SBATCH --output=logs/job_start.out
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/job_start.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/job_start.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=debug

module load python/anaconda-2021.05
python scripts/convGRU.py


