# Slurm job control

## Read-only inspection

Use both live and accounting views:

```bash
squeue -j JOB_ID -o '%.18i %.28j %.10T %.10M %R'
sacct -X -j JOB_ID --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End
```

Then inspect `logs/*-JOB_ID.out` and `logs/*-JOB_ID.err`. A missing live row commonly means the job completed, failed, timed out, or was cancelled; it does not mean the job never ran.

For training, verify progress from `training_history.json`, `last.pt`, and log epoch lines. Compare file timestamps with scheduler end time.

## Submission graph

The checked-in helpers create these dependencies:

```text
Stage 1: prepare -> train -> evaluate
Stage 2: prepare -> train -> evaluate
```

Commands, after explicit authorization:

```bash
bash submit_pipeline.sh
bash submit_stage2_pipeline.sh
```

Submitting the helpers creates all three jobs. If preparation already succeeded, submit only the needed downstream Slurm file and use a dependency only when appropriate.

## Resume

Stage 1:

```bash
MV_RESUME=/exact/path/last.pt sbatch slurm/02_train_mvswin.slurm
```

Stage 2:

```bash
MV_STAGE2_RESUME=/exact/path/last.pt sbatch slurm/06_train_stage2.slurm
```

Before submission, inspect the checkpoint and confirm it belongs to the same stage and run configuration.

## Cancellation

After explicit authorization, cancel only exact verified job IDs:

```bash
scancel JOB_ID [JOB_ID ...]
```

Report whether dependent jobs were also cancelled or remain pending. Never use broad user/account cancellation filters for this workflow.
