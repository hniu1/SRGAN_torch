#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_gcm
#SBATCH -o logs/mv_gcm-%j.out
#SBATCH -e logs/mv_gcm-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${MV_BASE_DIR:-$PWD}
DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/artifacts/data/daymet_mv_1980_1990}
RUN_DIR=${MV_RUN_DIR:-${BASE_DIR}/artifacts/runs/climateswin_v1}
GCM_DIR=${MV_GCM_DIR:-/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/gcm/bias-corrected}
OUTPUT=${MV_GCM_OUTPUT:-${RUN_DIR}/gcm/ACCESS-CM2_ssp585_1980_2019_0p25deg.nc}
mkdir -p "${BASE_DIR}/logs" "$(dirname "${OUTPUT}")"

cd "${BASE_DIR}"
srun python -u pipeline_04_downscale_gcm.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output "${OUTPUT}" \
  --input "tmin=${GCM_DIR}/tmin_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_198001-201912_1deg_NA_BC.nc" \
  --input "tmax=${GCM_DIR}/tmax_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_198001-201912_1deg_NA_BC.nc" \
  --input "prcp=${GCM_DIR}/pr_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_198001-201912_1deg_NA_BC.nc" \
  --start-date 1980-01-01 \
  --batch-size 4 \
  --format netcdf \
  --amp \
  --enforce-temperature-order

