#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_s2_prepare
#SBATCH -o logs/mv_s2_prepare-%j.out
#SBATCH -e logs/mv_s2_prepare-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate "${REFINE_ENV:-/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm}"

BASE_DIR=${MV_BASE_DIR:-$PWD}
STAGE=${MV_STAGE:-1}
DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/daymet/prepared/stage${STAGE}}
SOURCE_ROOT=${MV_SOURCE_ROOT:-/lustre/orion/proj-shared/cli138/dr6/Daymet1km/daymet_TVA_latlon}
: "${MV_DEM_LR:?Set MV_DEM_LR to the aligned input-grid DEM}"
: "${MV_DEM_HR:?Set MV_DEM_HR to the aligned target-grid DEM}"
mkdir -p "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_01_prepare.py \
  --output-dir "${DATA_DIR}" \
  --stage "${STAGE}" --data-root "${SOURCE_ROOT}" \
  --dem-lr "${MV_DEM_LR}" --dem-hr "${MV_DEM_HR}" \
  --variables tmin tmax prcp \
  --train-start "${MV_TRAIN_START:-1982}" --train-end "${MV_TRAIN_END:-2011}" \
  --val-start "${MV_VAL_START:-2011}" --val-end "${MV_VAL_END:-2016}" \
  --test-start "${MV_TEST_START:-2016}" --test-end "${MV_TEST_END:-2021}"
