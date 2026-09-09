#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_s2_train
#SBATCH -o logs/mv_s2_train-%j.out
#SBATCH -e logs/mv_s2_train-%j.err
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
conda activate "${REFINE_ENV:-/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm}"

BASE_DIR=${MV_BASE_DIR:-$PWD}
DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/artifacts/data/daymet_training}
RUN_DIR=${MV_RUN_DIR:-${BASE_DIR}/artifacts/runs/refine_6x}
RESUME_ARGS=()
if [[ -n "${MV_RESUME:-}" ]]; then
  if [[ ! -f "${MV_RESUME}" ]]; then
    echo "Stage-2 resume checkpoint not found: ${MV_RESUME}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${MV_RESUME}")

fi
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-3444}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export MIOPEN_USER_DB_PATH=/tmp/miopen-refine-stage2-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}" "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_02_train.py \
  --data-dir "${DATA_DIR}" \
  --run-dir "${RUN_DIR}" \
  "${RESUME_ARGS[@]}" \
  --epochs 100 \
  --batch-size 4 \
  --patches-per-day 8 \
  --validation-patches-per-day 4 \
  --core-size 8 \
  --halo 2 \
  --embed-dim 96 \
  --num-groups 6 \
  --blocks-per-group 6 \
  --num-heads 6 \
  --window-size 8 \
  --learning-rate 2e-4 \
  --warmup-epochs 5 \
  --gradient-clip 1.0 \
  --amp
