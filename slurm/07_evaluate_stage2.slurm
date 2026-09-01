#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_s2_eval
#SBATCH -o logs/mv_s2_eval-%j.out
#SBATCH -e logs/mv_s2_eval-%j.err
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
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
export MIOPEN_USER_DB_PATH=/tmp/miopen-mveval-s2-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}" "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_07_evaluate_stage2.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output-dir "${RUN_DIR}/test_1990" \
  --split test \
  --batch-size 1 \
  --amp \
  --no-save-predictions \
  --enforce-temperature-order
