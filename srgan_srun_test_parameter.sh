#!/bin/bash
#SBATCH -A cli138
#SBATCH -J SRGAN_S1_PARAM
#SBATCH -o logs/srgan_s1_param-%j.out
#SBATCH -e logs/srgan_s1_param-%j.err
#SBATCH -p extended
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH -t 12:00:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-3442}

# Needed to bypass MIOpen disk I/O issues
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# -----------------------------
# Sweep-friendly parameters
# Override with:
# sbatch --export=ALL,VAR=prcp,VERSION=dy_v0.1_a1,W1_FN1=1e-5,W2_FN2=1e3,CONTENT_LOSS=smoothl1 srgan_srun_test_parameter.sh
# -----------------------------
MODE=${MODE:-train}
VAR=${VAR:-prcp}
VERSION=${VERSION:-dy_v0.1_test_param}

BATCH_SIZE=${BATCH_SIZE:-4}
N_EPOCH=${N_EPOCH:-50}
N_EPOCH_INIT=${N_EPOCH_INIT:-20}

W1_FN1=${W1_FN1:-1e-5}
W2_FN2=${W2_FN2:-1e3}

AMP_FLAG=${AMP_FLAG:-0}
INITIAL_TRAINING=${INITIAL_TRAINING:-0}

AMP_ARG=""
if [[ "${AMP_FLAG}" == "1" ]]; then
  AMP_ARG="--amp"
fi

INIT_ARG=""
if [[ "${INITIAL_TRAINING}" == "1" ]]; then
  INIT_ARG="--initial-training"
fi

echo "=== Run config ==="
echo "MODE=${MODE}"
echo "VAR=${VAR}"
echo "VERSION=${VERSION}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "N_EPOCH=${N_EPOCH}"
echo "N_EPOCH_INIT=${N_EPOCH_INIT}"
echo "W1_FN1=${W1_FN1}"
echo "W2_FN2=${W2_FN2}"
echo "AMP_FLAG=${AMP_FLAG}"
echo "INITIAL_TRAINING=${INITIAL_TRAINING}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "=================="

srun --kill-on-bad-exit=1 python3 -u train_daily_frontier_srun_test_parameter.py \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --mode "${MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --version "${VERSION}" \
  --base-dir /lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr \
  --var "${VAR}" \
  --n-epoch "${N_EPOCH}" \
  --n-epoch-init "${N_EPOCH_INIT}" \
  --w1-fn1 "${W1_FN1}" \
  --w2-fn2 "${W2_FN2}" \
  ${AMP_ARG} \
  ${INIT_ARG}
