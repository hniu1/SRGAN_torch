#!/bin/bash
#SBATCH -A cli138
#SBATCH -J V_0.6
#SBATCH -o logs/srgan_0.6-%j.out
#SBATCH -e logs/srgan_0.6-%j.err
#SBATCH -p extended          # partition (queue)

#SBATCH -N 2                       # number of nodes
#SBATCH --ntasks-per-node=4        # 1 task per GPU
#SBATCH --gpus-per-task=1          # exactly 1 GPU per rank
#SBATCH --cpus-per-task=6          # CPU cores per rank
#SBATCH -t 12:00:00                # HARD limit for batch on cli138, no hard limit for extended

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

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


srun \
  python3 -u train_daily_frontier_srun_2nd.py \
    --master_addr=$MASTER_ADDR \
    --master_port=3442 \
    --mode train \
    --batch-size 4 \
    --version dy_v0.6 \
    --base-dir /lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr \
    --var tmax \
    --w1-fn1 1e-5 \
    --w2-fn2 1e3 
    # --initial-training
    # --master_port=3442
    # --read-raw
        # --var prcp \
        # --gpu-bind=closest --export=ALL,HIP_VISIBLE_DEVICES=$SLURM_LOCALID

      # for --w1-fn1 1e-4 for tmin and prcp
      # --w2-fn2 1e4 for tmin and prcp
