#!/bin/bash
#SBATCH -A cli138
#SBATCH -J npy2nc_dims
#SBATCH -o logs/npy2nc_dims-%j.out
#SBATCH -e logs/npy2nc_dims-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr
export OUTPUT_DIR_ROOT=${OUTPUT_DIR_ROOT:-$BASE_DIR/gcm_ds/ssp585_yearly_nc_0416deg_dims}

mkdir -p "$BASE_DIR/logs"
mkdir -p "$OUTPUT_DIR_ROOT"

echo "============================================================"
echo "NPY TO YEARLY NETCDF (WITH DIMS)"
echo "============================================================"
echo "OUTPUT_DIR_ROOT = $OUTPUT_DIR_ROOT"
echo "============================================================"

cd "$BASE_DIR"
python -u gcm_ds_npy_to_netcdf_save_yearly.py
