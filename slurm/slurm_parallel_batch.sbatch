#!/bin/bash

#SBATCH --job-name=job_%a
#SBATCH --ntasks=8
#SBATCH --output=logs/job_%a.out
#SBATCH --array=1,2,3,4,5,6,7,8
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/job_%a.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/job_%a.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=debug
#SBATCH --exclusive

module load python/anaconda-2021.05
python scripts/convGRU.py


