#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_s2_apply
#SBATCH -o logs/mv_s2_apply-%j.out
#SBATCH -e logs/mv_s2_apply-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate "${REFINE_ENV:-/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm}"

BASE_DIR=${MV_BASE_DIR:-$PWD}
DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/daymet/prepared}
RUN_DIR=${MV_RUN_DIR:-${BASE_DIR}/artifacts/runs/refine_6x}
MV_TMIN_INPUT=${MV_TMIN_INPUT:-daymet/data/Daymet_ERA5_tmin_dy_1990_0p25deg.nc}
MV_TMAX_INPUT=${MV_TMAX_INPUT:-daymet/data/Daymet_ERA5_tmax_dy_1990_0p25deg.nc}
MV_PRCP_INPUT=${MV_PRCP_INPUT:-daymet/data/Daymet_ERA5_prcp_dy_1990_0p25deg.nc}
MV_OUTPUT=${MV_OUTPUT:-artifacts/demo/inference-${SLURM_JOB_ID}.nc}
export MIOPEN_USER_DB_PATH=/tmp/miopen-mvapply-s2-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}" "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_04_infer.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint "${MV_CHECKPOINT:-${BASE_DIR}/checkpoints/refine_6x.pt}" \
  --input "tmin=${MV_TMIN_INPUT}" \
  --input "tmax=${MV_TMAX_INPUT}" \
  --input "prcp=${MV_PRCP_INPUT}" \
  --output "${MV_OUTPUT}" \
  --start-date 1990-01-01 \
  --start-index "${MV_START_INDEX:-0}" \
  --end-index "${MV_END_INDEX:-1}" \
  --format netcdf \
  --batch-size 1 \
  --amp \
  --enforce-temperature-order
