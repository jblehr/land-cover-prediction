#!/bin/bash

#SBATCH --job-name=job_%a
#SBATCH --ntasks=2
#SBATCH --output=logs/job_%a.out
#SBATCH --array=1,2
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_%a.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_%a.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro

module load python/anaconda-2021.05
python scripts/convGRU.py


