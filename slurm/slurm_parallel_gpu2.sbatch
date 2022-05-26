#!/bin/bash

#SBATCH --job-name=convgru
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/midway_train_convgru.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/midway_train_convgru.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --nodes=1           # 1 GPU node
#SBATCH --exclusive

module load cuda/6.5
module load python/anaconda-2021.05

python3 scripts/convGRU.py

