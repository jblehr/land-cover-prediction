#!/bin/bash

#SBATCH --job-name=mtrain
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/bigmem_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/bigmem_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=bigmem2
#SBATCH --ntasks=1          # 1 CPU core to drive GPU

module load cuda/6.5
module load python/anaconda-2021.05

python3 scripts/convGRU.py --optuna


