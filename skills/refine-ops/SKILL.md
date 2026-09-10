---
name: refine-ops
description: Prepare data, train or resume REFINE models, and operate the repository's Slurm jobs on Frontier. Use for scheduler status, logs, dependencies, resource configuration, checkpoint recovery, and pipeline submissions. Do not use for scientific evaluation interpretation or ordinary inference command construction.
---

# REFINE Operations

Use repository-local `daymet/data/`, `daymet/dem/`, and `daymet/prepared/`
for demo data, with `checkpoints/refine_6x.pt` for the pretrained model.
The same prepared index contains training, validation, and test splits.
Do not use original `proj-shared` data paths from provenance records as runtime
inputs or fallbacks. If assets are missing, obtain the complete demo archive.


Operate the single-stage 6× REFINE workflow conservatively on Frontier. Read-only scheduler and filesystem inspection may proceed directly; state-changing scheduler actions require explicit user authorization.

## Workflow

1. Read [references/frontier.md](references/frontier.md) and confirm the repository, allocation, environment, and artifact paths.
2. Determine whether the request is data preparation, fresh training, resume training, evaluation, inference, plotting, or status investigation.
3. For training, read [references/training.md](references/training.md). Verify the manifest and intended checkpoint behavior before constructing a submission.
4. For scheduler actions, read [references/job-control.md](references/job-control.md).
5. Inspect active and historical jobs with the bundled read-only helper:

   ```bash
   python skills/refine-ops/scripts/job_status.py JOB_ID [JOB_ID ...]
   ```

6. Inspect the corresponding `logs/*-JOB_ID.out` and `.err`, checkpoint metadata, and training history before diagnosing a failure or deciding to resume.
7. Before scheduler mutations, verify that existing user authorization covers the command and targets; ask only if it does not.
8. After submission, return job IDs, dependencies, log paths, and the success condition to monitor.

## Guardrails

- Never resubmit solely because a job is absent from `squeue`; query `sacct` first.
- Never resume from a checkpoint without validating the file, stage, model configuration, epoch, and optimizer state.
- Avoid launching a full pipeline if completed preparation artifacts already match the intended manifest.
- Do not overwrite a run directory containing results unless the user explicitly chooses that outcome.
- Do not cancel pending/running jobs without explicit authorization and exact job IDs.
- Keep demo test data separate from training/validation data and preserve the released checkpoint.
- A zero exit code establishes operational completion, not model quality. Use `$refine-evaluation` for scientific assessment.

## Output

Return an operations record with action, command, job IDs and dependency graph, resource request, input/run paths, checkpoint mode, log paths, current state, and next verification step.
