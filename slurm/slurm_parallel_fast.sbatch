#!/bin/bash

#SBATCH --job-name=logis
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/logis.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/logis.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=fast

module load python/anaconda-2021.05
python scripts/baseline_logistic.py


