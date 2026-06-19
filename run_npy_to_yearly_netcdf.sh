#!/bin/bash
#SBATCH -A cli138
#SBATCH -J npy2nc
#SBATCH -o ./logs/npy2nc-%j.out
#SBATCH -e ./logs/npy2nc-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00

set -euo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

BASE_DIR=${BASE_DIR:-/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr}

VAR=${VAR:-pr}
MODEL=${MODEL:-ACCESS-CM2}
SCENARIO=${SCENARIO:-ssp585}
ENSEMBLE=${ENSEMBLE:-r1i1p1f1}
GRID=${GRID:-gn}

DOWNSCALE_VERSION=${DOWNSCALE_VERSION:-0.2}
YEAR_START=${YEAR_START:-2020}
YEAR_END=${YEAR_END:-2060}

FILE_PATTERN=${FILE_PATTERN:-legacy}

NPY_FILE=${NPY_FILE:-${BASE_DIR}/gcm_ds/${DOWNSCALE_VERSION}/${VAR}/${MODEL}/y_pred_4.npy}
OUTDIR=${OUTDIR:-${BASE_DIR}/gcm_ds/${SCENARIO}_yearly_nc_before_BC_0416deg/${VAR}/${MODEL}}

mkdir -p ./logs "$OUTDIR"

echo "============================================================"
echo "NPY TO YEARLY NETCDF CONVERSION"
echo "============================================================"
echo "VAR               = $VAR"
echo "MODEL             = $MODEL"
echo "SCENARIO          = $SCENARIO"
echo "ENSEMBLE          = $ENSEMBLE"
echo "GRID              = $GRID"
echo "DOWNSCALE_VERSION = $DOWNSCALE_VERSION"
echo "YEAR_START        = $YEAR_START"
echo "YEAR_END(excl)    = $YEAR_END"
echo "NPY_FILE          = $NPY_FILE"
echo "OUTDIR            = $OUTDIR"
echo "FILE_PATTERN      = $FILE_PATTERN"
echo "============================================================"

if [[ ! -f "$NPY_FILE" ]]; then
  echo "[ERROR] Missing input NPY file: $NPY_FILE"
  exit 1
fi

python3 -u "${BASE_DIR}/npy_to_yearly_netcdf.py" \
  --var "$VAR" \
  --gcm "$MODEL" \
  --scenario "$SCENARIO" \
  --ensemble "$ENSEMBLE" \
  --grid "$GRID" \
  --downscale-version "$DOWNSCALE_VERSION" \
  --year-start "$YEAR_START" \
  --year-end "$YEAR_END" \
  --npy-file "$NPY_FILE" \
  --output-dir "$OUTDIR" \
  --file-pattern "$FILE_PATTERN"

echo "[ALL DONE]"
