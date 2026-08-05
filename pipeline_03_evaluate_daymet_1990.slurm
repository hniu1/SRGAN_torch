#!/bin/bash
#SBATCH -A cli138
#SBATCH -J daymet1990_eval
#SBATCH -o logs/daymet1990_eval-%j.out
#SBATCH -e logs/daymet1990_eval-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH -t 01:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${SRGAN_BASE_DIR:-$PWD}
VERSION=${SRGAN_VERSION:-tmax_stage1_patch_pixelshuffle_10yr}
DATA_VERSION=${SRGAN_DATA_VERSION:-tmax_stage1_patch_10yr_data}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-daymet-${USER}-${SLURM_JOB_ID}}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/tmp/miopen-daymet-${USER}-${SLURM_JOB_ID}}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-${MIOPEN_USER_DB_PATH}}
mkdir -p "${MPLCONFIGDIR}" "${MIOPEN_USER_DB_PATH}" logs

python -u pipeline_03_evaluate_daymet_1990.py \
  --base-dir "${BASE_DIR}" \
  --version "${VERSION}" \
  --data-version "${DATA_VERSION}" \
  --checkpoint g_init.pth \
  --var tmax \
  --year 1990 \
  --batch-size 32 \
  --device auto \
  --save-prediction
