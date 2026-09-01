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
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${MV_BASE_DIR:-$PWD}
DATA_DIR=${MV_STAGE2_DATA_DIR:-${BASE_DIR}/artifacts/data/daymet_mv_stage2_1980_1990}
RUN_DIR=${MV_STAGE2_RUN_DIR:-${BASE_DIR}/artifacts/runs/climateswin_stage2_v1}
RESUME_ARGS=()
if [[ -n "${MV_STAGE2_RESUME:-}" ]]; then
  if [[ ! -f "${MV_STAGE2_RESUME}" ]]; then
    echo "Stage-2 resume checkpoint not found: ${MV_STAGE2_RESUME}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${MV_STAGE2_RESUME}")
else
  STAGE1_CHECKPOINT=${MV_STAGE1_CHECKPOINT:-${BASE_DIR}/artifacts/runs/climateswin_v1/best.pt}
  if [[ -f "${STAGE1_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--init-backbone "${STAGE1_CHECKPOINT}")
  else
    echo "Stage-1 checkpoint not found; Stage 2 will initialize from scratch: ${STAGE1_CHECKPOINT}" >&2
  fi
fi
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-3444}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export MIOPEN_USER_DB_PATH=/tmp/miopen-mvswin-s2-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}" "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_06_train_stage2.py \
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
