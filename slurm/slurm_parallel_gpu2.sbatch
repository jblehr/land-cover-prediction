#!/bin/bash

#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=gpu2
#SBATCH --exclusive

module load cuda/6.5
module load python/anaconda-2020.02

python scripts/convGRU.py --optuna


