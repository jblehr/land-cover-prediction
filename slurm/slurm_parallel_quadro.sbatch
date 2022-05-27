#!/bin/bash

#SBATCH --job-name=train_bptt
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro
#SBATCH --exclusive
#SBATCH --gres=gpu:1

cuda_version=11.1
export CUDA_HOME=/usr/local/cuda-${cuda_version}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:/usr/local/cuda/bin

# python scripts/convGRU.py
python scripts/cluster_cuda_check.py
