#!/bin/bash

#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro

module load python/anaconda-2021.05
python scripts/convGRU.py


