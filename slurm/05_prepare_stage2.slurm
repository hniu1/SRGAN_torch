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
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${MV_BASE_DIR:-$PWD}
STAGE1_DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/artifacts/data/daymet_mv_1980_1990}
STAGE2_DATA_DIR=${MV_STAGE2_DATA_DIR:-${BASE_DIR}/artifacts/data/daymet_mv_stage2_1980_1990}
mkdir -p "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u pipeline_05_prepare_stage2.py \
  --output-dir "${STAGE2_DATA_DIR}" \
  --stage1-manifest "${STAGE1_DATA_DIR}/manifest.json" \
  --variables tmin tmax prcp \
  --train-start 1980 --train-end 1988 \
  --val-start 1988 --val-end 1990 \
  --test-start 1990 --test-end 1991
