# Operations and permissions

## May proceed read-only

- Read manifests, configuration, logs, metrics, and checkpoint metadata.
- Check file existence, size, timestamps, hashes, dimensions, variables, and units.
- Run input validators and result summarizers.
- Query exact job IDs with `squeue` and `sacct`.
- Construct commands and estimate resource/storage needs.

## Require explicit approval immediately before execution

- Submit any Slurm job with `sbatch`.
- Cancel jobs with `scancel`.
- Start a long local/GPU inference or training run.
- Overwrite existing data, checkpoints, predictions, metrics, or plots.
- Delete/move artifacts or alter the model registry.

Approval should name the exact action, job IDs or output paths, and material resource impact. Prior approval for one job does not authorize a later resubmission or cancellation.

## Never do implicitly

- Broadly cancel by user/account/partition.
- Choose a new scientific missing-data or clipping policy.
- Regrid or convert calendars without recording the method.
- Treat a completed job as a validated model.
- Publish or register a checkpoint based only on training loss.
- Present Stage 2 evaluation using true 0.25-degree Daymet inputs as cascade performance using Stage 1 predictions.

## Reproducibility record

Record repository revision and dirty state, effective environment/path overrides, manifests and checkpoint hashes, full command, scheduler job ID, time interval, output checksum/size, validation warnings, and evaluation identity.
