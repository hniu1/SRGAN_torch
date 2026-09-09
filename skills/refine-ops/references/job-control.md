# Frontier job control

From the repository root, create logs/ before submission. Use the existing user
authorization when it covers the action; otherwise obtain it before submitting,
cancelling or overwriting results.

- Inference: `sbatch slurm/04_infer.slurm`
- Training chain: `bash submit_pipeline.sh` (requires training years staged first)
- Resume: `MV_RESUME=/exact/path/last.pt sbatch slurm/02_train.slurm`
- Status: `python skills/refine-ops/scripts/job_status.py JOB_ID`

Inspect squeue and sacct before deciding to resubmit. Report job IDs, dependencies,
log paths, effective asset paths and success criteria. Never cancel by broad
user/account filters. Preserve existing experiment outputs.
