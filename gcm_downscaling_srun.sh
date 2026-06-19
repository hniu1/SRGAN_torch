#!/bin/bash
#SBATCH -A cli138
#SBATCH -J ds_03_test
#SBATCH -o logs/ds_03_test-%j.out
#SBATCH -e logs/ds_03_test-%j.err
#SBATCH -p extended
#SBATCH -N 1       
#SBATCH -t 10:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

# conda init bash
# source ~/.bashrc
# conda activate srgan

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

python -u gcm_downscaling.py