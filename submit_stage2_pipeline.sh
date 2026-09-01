#!/bin/bash
set -euo pipefail

mkdir -p logs
prepare_job=$(sbatch --parsable slurm/05_prepare_stage2.slurm)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" slurm/06_train_stage2.slurm)
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" slurm/07_evaluate_stage2.slurm)

echo "stage2_prepare=${prepare_job} stage2_train=${train_job} stage2_evaluate=${eval_job}"
