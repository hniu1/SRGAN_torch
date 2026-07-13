#!/bin/bash
#SBATCH -A cli138
#SBATCH -J patch1_test
#SBATCH -o logs/stage1_patch-%j.out
#SBATCH -e logs/stage1_patch-%j.err
#SBATCH -p extended          # partition (queue)

#SBATCH -N 2                       # number of nodes
#SBATCH --ntasks-per-node=4        # 1 task per GPU
#SBATCH --gpus-per-task=1          # exactly 1 GPU per rank
#SBATCH --cpus-per-task=6          # CPU cores per rank
#SBATCH -t 6:00:00                # HARD limit for batch on cli138, no hard limit for extended

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${SRGAN_BASE_DIR:-$PWD}

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}


VERSION=${SRGAN_VERSION:-tmax_stage1_patch_test}
VAR=${SRGAN_VAR:-tmax}

srun \
  python3 -u train_daily_frontier_srun.py \
    --master_addr=$MASTER_ADDR \
    --master_port=3442 \
    --mode train \
    --batch-size 64 \
    --version ${VERSION} \
    --base-dir ${BASE_DIR} \
    --var ${VAR} \
    --patch-training \
    --lr-patch-size 8 \
    --scale-factor 4 \
    --patches-per-image 4 \
    --w1-fn1 1e-5 \
    --w2-fn2 1e3