#!/bin/bash
#SBATCH -A cli138
#SBATCH -J srgan-yearly-test
#SBATCH -o ./logs/srgan-yearly-test-%j.out
#SBATCH -e ./logs/srgan-yearly-test-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH -t 01:00:00

set -euo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1

export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
rm -rf "$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

BASE_DIR=/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/gcm_ds/output_test/pr/ACCESS-CM2"

echo "============================================================"
echo "SRGAN YEARLY INFERENCE -- TEST RUN (1 year: 1980)"
echo "============================================================"

srun python3 -u "$BASE_DIR/gcm_downscaling_dims_yearly.py" \
  --var          pr \
  --gcm          ACCESS-CM2 \
  --scenario     ssp585 \
  --ensemble     r1i1p1f1 \
  --grid         gn \
  --version-lr   dy_v0.3 \
  --version-hr   dy_v0.4 \
  --downscale-version test \
  --year-start   1980 \
  --year-end     1981 \
  --gcm-file     /lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/gcm/pr_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_198001-201912_1deg_NA_BC.nc \
  --base-dir     "$BASE_DIR" \
  --output-dir   "$BASE_DIR/gcm_ds/output_test/pr/ACCESS-CM2" \
  --elevation \
  --batch-size-lr 4 \
  --batch-size-hr 2 \
  --file-pattern  legacy

echo "[ALL DONE]"
