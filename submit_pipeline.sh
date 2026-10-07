#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export MV_BASE_DIR=${MV_BASE_DIR:-$PWD}
export MV_STAGE=${MV_STAGE:-1}
export MV_DATA_DIR=${MV_DATA_DIR:-${MV_BASE_DIR}/daymet/prepared/stage${MV_STAGE}}
export MV_RUN_DIR=${MV_RUN_DIR:-${MV_BASE_DIR}/artifacts/runs/refine_stage${MV_STAGE}_5x}
: "${MV_DEM_LR:?Set MV_DEM_LR before submitting preparation}"
: "${MV_DEM_HR:?Set MV_DEM_HR before submitting preparation}"
prepare=$(sbatch --parsable slurm/01_prepare.slurm)
train=$(sbatch --parsable --dependency=afterok:${prepare} slurm/02_train.slurm)
MV_CHECKPOINT="${MV_RUN_DIR}/best.pt" sbatch --dependency=afterok:${train} slurm/03_evaluate.slurm
