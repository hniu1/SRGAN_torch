#!/bin/bash
set -euo pipefail

mkdir -p logs

prepare_job=$(sbatch --parsable slurm/01_prepare_stage1.slurm)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" slurm/02_train_stage1.slurm)
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" slurm/03_evaluate_stage1.slurm)

echo "prepare=${prepare_job} train=${train_job} evaluate=${eval_job}"
