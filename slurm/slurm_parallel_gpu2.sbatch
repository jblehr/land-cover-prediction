#!/bin/bash

#SBATCH --job-name=mtrain
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/midway_train_%a.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/midway_train_%a.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --nodes=1           # 1 GPU node
#SBATCH --array=1,2

module load cuda/6.5
module load python/anaconda-2021.05

python3 scripts/convGRU.py --optuna


