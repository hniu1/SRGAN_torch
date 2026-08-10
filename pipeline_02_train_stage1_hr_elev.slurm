#!/bin/bash
#SBATCH -A cli138
#SBATCH -J patch_hr_elev
#SBATCH -o logs/stage1_hr_elev-%j.out
#SBATCH -e logs/stage1_hr_elev-%j.err
#SBATCH -p extended
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH -t 6:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${SRGAN_BASE_DIR:-$PWD}
VERSION=${SRGAN_VERSION:-tmax_stage1_patch_hr_elev_stable_10yr}
DATA_VERSION=${SRGAN_DATA_VERSION:-tmax_stage1_patch_hr_elev_stable_10yr}
VAR=${SRGAN_VAR:-tmax}

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1
export MASTER_ADDR=$(hostname -i)
export MIOPEN_USER_DB_PATH=/tmp/miopen-hr-elev-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}" logs

srun python3 -u pipeline_02_train_stage1_hr_elev.py \
  --master_addr="${MASTER_ADDR}" \
  --master_port=3443 \
  --mode train \
  --batch-size 32 \
  --version "${VERSION}" \
  --data-version "${DATA_VERSION}" \
  --base-dir "${BASE_DIR}" \
  --var "${VAR}" \
  --generator patch_hr_elev \
  --patch-training \
  --lr-patch-size 16 \
  --scale-factor 4 \
  --patches-per-image 64 \
  --initial-training \
  --n-epoch-init 50 \
  --n-epoch 50 \
  --w1-fn1 1e-4 \
  --w2-fn2 1.0 \
  --w-absolute 0.1 \
  --w-gradient 0.1 \
  --lr-init 5e-5 \
  --lr-generator 2e-5 \
  --lr-discriminator 1e-5 \
  --gradient-clip 1.0 \
  --d-train-threshold 0.6 \
  --early-stop-patience-init 15 \
  --early-stop-patience-adv 20
