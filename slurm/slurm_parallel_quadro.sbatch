#!/bin/bash

#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro
#SBATCH --exclusive

module load cuda/10.0
python scripts/convGRU.py --optuna


