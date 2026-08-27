#!/bin/bash
set -euo pipefail

mkdir -p logs

prepare_job=$(sbatch --parsable slurm/01_prepare_multivariable.slurm)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" slurm/02_train_mvswin.slurm)
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" slurm/03_evaluate_mvswin.slurm)

echo "prepare=${prepare_job} train=${train_job} evaluate=${eval_job}"
