#!/bin/bash
#SBATCH -A cli138
#SBATCH -J eval_hr_elev
#SBATCH -o logs/daymet1990_hr_elev-%j.out
#SBATCH -e logs/daymet1990_hr_elev-%j.err
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
VERSION=${SRGAN_VERSION:-tmax_stage1_patch_hr_elev_deep_10yr}
DATA_VERSION=${SRGAN_DATA_VERSION:-tmax_stage1_patch_hr_elev_deep_10yr}
export MPLCONFIGDIR=/tmp/mpl-hr-elev-${USER}-${SLURM_JOB_ID}
export MIOPEN_USER_DB_PATH=/tmp/miopen-eval-hr-elev-${USER}-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MPLCONFIGDIR}" "${MIOPEN_USER_DB_PATH}" logs

python -u pipeline_03_evaluate_daymet_1990.py \
  --base-dir "${BASE_DIR}" \
  --version "${VERSION}" \
  --data-version "${DATA_VERSION}" \
  --generator patch_hr_elev \
  --checkpoint g_init.pth \
  --var tmax \
  --year 1990 \
  --batch-size 16 \
  --device auto \
  --save-prediction
