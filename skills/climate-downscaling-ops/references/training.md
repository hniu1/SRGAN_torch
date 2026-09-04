# Preparation and training

## Stage 1: 1 degree to 0.25 degree

Pipeline scripts:

1. `pipeline_01_prepare_multivariable.py`
2. `pipeline_02_train_mvswin.py`
3. `pipeline_03_evaluate_mvswin.py`

`submit_pipeline.sh` submits the matching Slurm scripts with `afterok` dependencies. Current chronological split is training 1980-1987, validation 1988-1989, and test 1990 because the year-end arguments are exclusive.

Training requests 2 nodes, 4 GPU tasks per node, 6 CPUs per task, and 12 hours in `extended`. Defaults are 100 epochs, batch size 4 per process, 32 patches/day, 8 validation patches/day, core 16, halo 4, embed dimension 96, 6 groups, 6 blocks/group, 6 heads, and window 8.

Set `MV_RESUME=/exact/checkpoint.pt` to resume. Confirm the checkpoint's training state before submission.

## Stage 2: 0.25 degree to 1/24 degree

Pipeline scripts:

1. `pipeline_05_prepare_stage2.py`
2. `pipeline_06_train_stage2.py`
3. `pipeline_07_evaluate_stage2.py`

`submit_stage2_pipeline.sh` submits these with `afterok` dependencies and the same chronological split. Training uses the same model width/depth and distributed resources, with 8 patches/day, 4 validation patches/day, core 8, and halo 2.

Checkpoint selection order:

1. If `MV_STAGE2_RESUME` is set, resume full Stage 2 state from that exact checkpoint.
2. Otherwise initialize the compatible backbone from `MV_STAGE1_CHECKPOINT` or the default Stage 1 `best.pt`.
3. If the Stage 1 checkpoint is missing, Stage 2 initializes from scratch and emits a warning.

Backbone initialization is not resume training: optimizer, scheduler, and Stage 2 upsampling head start fresh.

## Pre-submission checks

- Parse the manifest and verify years, variables, shapes, units, and patch count.
- Confirm no active job is already writing to the data/run directory.
- Confirm sufficient storage for checkpoints, logs, and optional predictions.
- For resume, inspect `last.pt` and training history; do not assume `best.pt` contains resumable optimizer state.
- Review resource/time limits against the intended workload.
- Record code revision and dirty-worktree state for reproducibility.
