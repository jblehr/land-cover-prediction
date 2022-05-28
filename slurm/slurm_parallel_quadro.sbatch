#!/bin/bash

#SBATCH --job-name=logis
#SBATCH --output=/home/npg/land-cover-prediction/slurm/logs/quad_train_logis.out
#SBATCH --error=/home/npg/land-cover-prediction/slurm/logs/quad_train_logis.err
#SBATCH --chdir=/home/npg/land-cover-prediction
#SBATCH --partition=quadro
#SBATCH --exclusive

cuda_version=11.1
export CUDA_HOME=/usr/local/cuda-${cuda_version}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:/usr/local/cuda/bin

# python scripts/convGRU.py
# python scripts/cluster_cuda_check.py
python scripts/baseline_logistic.py
