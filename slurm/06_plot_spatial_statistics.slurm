#!/bin/bash
#SBATCH -A cli138
#SBATCH -J mv_s2_spatial
#SBATCH -o logs/mv_s2_spatial-%j.out
#SBATCH -e logs/mv_s2_spatial-%j.err
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
conda activate "${REFINE_ENV:-/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm}"

BASE_DIR=${MV_BASE_DIR:-$PWD}
DATA_DIR=${MV_DATA_DIR:-${BASE_DIR}/daymet/prepared}
RUN_DIR=${MV_RUN_DIR:-${BASE_DIR}/artifacts/runs/refine_6x}
export MPLCONFIGDIR=/tmp/matplotlib-mvspatial-s2-${USER}-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${BASE_DIR}/logs"

cd "${BASE_DIR}"
srun python -u utility_plot_spatial_statistics.py \
  --data-dir "${DATA_DIR}" \
  --evaluation-dir "${RUN_DIR}/test_1990" \
  --split test \
  --tile-rows 21
