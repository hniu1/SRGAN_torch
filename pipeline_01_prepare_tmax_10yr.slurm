#!/bin/bash
#SBATCH -A cli138
#SBATCH -J data_tmax_10yr
#SBATCH -o logs/data_tmax_10yr-%j.out
#SBATCH -e logs/data_tmax_10yr-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${SRGAN_BASE_DIR:-$PWD}
DATA_VERSION=${SRGAN_DATA_VERSION:-tmax_stage1_patch_10yr_data}

mkdir -p logs

srun python -u pipeline_01_prepare_daymet.py \
  --base-dir "${BASE_DIR}" \
  --version "${DATA_VERSION}" \
  --var tmax \
  --year-start 1980 \
  --year-end 1990
