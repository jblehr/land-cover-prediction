#!/bin/bash

#SBATCH --job-name=job_%a
#SBATCH --ntasks=1
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_%a.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_%a.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro

module load python/anaconda-2021.05
python scripts/convGRU.py


