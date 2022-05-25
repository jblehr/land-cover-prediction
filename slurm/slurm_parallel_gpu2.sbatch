#!/bin/bash

#SBATCH --job-name=mtrain
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=gpu2
#SBATCH --ntasks=1          # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --nodes=1           # 1 GPU node

module load cuda/6.5
module load python/anaconda-2020.02

python3 scripts/convGRU.py --optuna


