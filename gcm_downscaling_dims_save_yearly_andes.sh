#!/bin/bash
#SBATCH -A cli138
#SBATCH -J dims_save_nc
#SBATCH -o ./logs/dims_save_nc-%j.out
#SBATCH -e ./logs/dims_save_nc-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00

conda init bash
source ~/.bashrc
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/srgan

BASE_DIR=${BASE_DIR:-/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr}

VAR=${VAR:-pr}
SCENARIO=${SCENARIO:-ssp585}
ENSEMBLE=${ENSEMBLE:-r1i1p1f1}
GRID=${GRID:-gn}

DEFAULT_MODELS=(
  ACCESS-CM2
  # BCC-CSM2-MR
  # CNRM-ESM2-1
  # EC-Earth3-CC
  # EC-Earth3-Veg
  # GFDL-CM4
  # MRI-ESM2-0
  # MPI-ESM1-2-HR
  # NorESM2-MM
  # TaiESM1
)

if [[ -n "${MODEL:-}" ]]; then
  MODEL_LIST=("$MODEL")
elif [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<< "$MODELS"
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

DOWNSCALE_VERSION=${DOWNSCALE_VERSION:-0.1}
YEAR_START=${YEAR_START:-1980}
YEAR_END=${YEAR_END:-2020}

FILE_PATTERN=${FILE_PATTERN:-legacy}
NC_COMPRESSION_LEVEL=${NC_COMPRESSION_LEVEL:-1}

case "$VAR" in
  pr)
    VERSION_LR=${VERSION_LR:-dy_v0.3}
    VERSION_HR=${VERSION_HR:-dy_v0.4}
    ;;
  tmax)
    VERSION_LR=${VERSION_LR:-dy_v0.5}
    VERSION_HR=${VERSION_HR:-dy_v0.6}
    ;;
  tmin)
    VERSION_LR=${VERSION_LR:-dy_v0.7}
    VERSION_HR=${VERSION_HR:-dy_v0.8}
    ;;
  *)
    echo "[ERROR] Unsupported VAR=$VAR"
    exit 1
    ;;
esac

if [[ ${#MODEL_LIST[@]} -gt 1 && ( -n "${NPY_FILE:-}" || -n "${OUTDIR:-}" ) ]]; then
  echo "[ERROR] NPY_FILE/OUTDIR overrides are only supported when running one MODEL."
  echo "        Use MODEL=ACCESS-CM2 sbatch $0, or unset NPY_FILE and OUTDIR for multi-model runs."
  exit 1
fi

mkdir -p ./logs

echo "============================================================"
echo "SRGAN DIMS YEARLY NETCDF (CPU SAVE-NC ONLY, ANDES)"
echo "============================================================"
echo "BASE_DIR          = $BASE_DIR"
echo "VAR               = $VAR"
echo "MODELS            = ${MODEL_LIST[*]}"
echo "SCENARIO          = $SCENARIO"
echo "ENSEMBLE          = $ENSEMBLE"
echo "GRID              = $GRID"
echo "DOWNSCALE_VERSION = $DOWNSCALE_VERSION"
echo "VERSION_LR        = $VERSION_LR"
echo "VERSION_HR        = $VERSION_HR"
echo "YEAR_START        = $YEAR_START"
echo "YEAR_END(excl)    = $YEAR_END"
echo "FILE_PATTERN      = $FILE_PATTERN"
echo "NC_COMPRESSION    = $NC_COMPRESSION_LEVEL"
echo "MODE              = save-nc"
echo "============================================================"

cd "$BASE_DIR"

for MODEL in "${MODEL_LIST[@]}"; do
  NPY_FILE_MODEL=${NPY_FILE:-${BASE_DIR}/gcm_ds/${DOWNSCALE_VERSION}/${VAR}/${MODEL}/y_pred_4.npy}
  OUTDIR_MODEL=${OUTDIR:-${BASE_DIR}/gcm_ds/${SCENARIO}_yearly_nc_before_BC_0416deg_dims/${VAR}/${MODEL}}

  mkdir -p "$OUTDIR_MODEL"

  echo "------------------------------------------------------------"
  echo "MODEL             = $MODEL"
  echo "NPY_FILE          = $NPY_FILE_MODEL"
  echo "OUTDIR            = $OUTDIR_MODEL"
  echo "------------------------------------------------------------"

  if [[ ! -f "$NPY_FILE_MODEL" ]]; then
    echo "[ERROR] Missing input NPY file: $NPY_FILE_MODEL"
    exit 1
  fi

  python -u gcm_downscaling_dims_yearly.py \
    --mode save-nc \
    --var "$VAR" \
    --gcm "$MODEL" \
    --scenario "$SCENARIO" \
    --ensemble "$ENSEMBLE" \
    --grid "$GRID" \
    --version-lr "$VERSION_LR" \
    --version-hr "$VERSION_HR" \
    --downscale-version "$DOWNSCALE_VERSION" \
    --year-start "$YEAR_START" \
    --year-end "$YEAR_END" \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUTDIR_MODEL" \
    --file-pattern "$FILE_PATTERN" \
    --nc-compression-level "$NC_COMPRESSION_LEVEL"
done

echo "[ALL DONE]"
