---
name: refine-downscaling
description: Validate Daymet inputs and run pretrained REFINE inference from 1/4 degree to 1/24 degree on Frontier. Use for the single-stage demo, checkpoint selection, inference commands and output interpretation.
---
# REFINE downscaling

Use repository-local `daymet/data/`, `daymet/dem/`, and `daymet/prepared/`
for demo data, with `checkpoints/refine_6x.pt` for the pretrained model.
The same prepared index contains training, validation, and test splits.
Do not use original `proj-shared` data paths from provenance records as runtime
inputs or fallbacks. If assets are missing, obtain the complete demo archive.


Locate the repository containing `pipeline_04_infer.py`. Run commands from that
root. The only route is 6×, with tmin/tmax/prcp together; `stage2` is its legacy
registry identifier.

1. Read [data contract](references/data-contract.md) and run `python scripts/check_demo.py`.
2. Read [checkpoint guidance](references/checkpoints.md) before selecting another model.
3. Read [inference commands](references/inference.md). Default to one day and a new output path.
4. Use the operations skill for an authorized Frontier submission; report the job ID.
5. Inspect output shape, interval, variables and run metadata before reporting success.

For experiments against truth, use `refine-evaluation`. For training or model
changes use `refine-ops` and its training reference. Preserve the released model
and record the code revision and checkpoint hash for comparisons.
