#!/bin/bash
#SBATCH -A cli138
#SBATCH -J gcm_stage1_cmp
#SBATCH -o logs/gcm_stage1_compare-%j.out
#SBATCH -e logs/gcm_stage1_compare-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH -t 02:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${SRGAN_BASE_DIR:-$PWD}
VERSION=${SRGAN_VERSION:-tmax_stage1_patch_pixelshuffle_10yr}
VAR=${SRGAN_VAR:-tmax}
N_DAYS=${SRGAN_N_DAYS:-366}

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig-${USER}-${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}" logs

python -u pipeline_04_diagnose_gcm.py \
  --base-dir "${BASE_DIR}" \
  --version "${VERSION}" \
  --var "${VAR}" \
  --n-days "${N_DAYS}" \
  --batch-size 16 \
  --device auto \
  --save-prediction
