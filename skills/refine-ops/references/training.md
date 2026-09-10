# Preparation and training: 1/4° to 1/24°

The existing full Stage 2 prepared index is
`daymet/prepared/`. Its manifest reads paired NetCDF
files from repository-local `daymet/data/`, DEMs from `daymet/dem/`, and frozen
normalization from `daymet/normalization.json`. The 1980–1990 input pairs are
present in this workspace. Verify the required years exist after any handoff;
the asset manifest inventories the original 1990 inputs and prepared arrays;
`check_demo.py` also checks source-file presence for every split. Use the existing prepared index
when it matches the experiment; to rebuild it:

```bash
python pipeline_01_prepare.py --data-root daymet/data --dem-root daymet/dem \
  --normalization-manifest daymet/normalization.json \
  --output-dir daymet/prepared
```

Splits are chronological: training 1980–1987, validation 1988–1989, test 1990.
Year-end arguments are exclusive. Preparation reuses frozen training-only
normalization for compatibility with the pretrained checkpoint. New variables
or a different climate/domain require explicitly fitting appropriate transforms
on training data; this preparation command does not fit new normalization.

After authorization, submit `slurm/02_train.slurm` with the prepared training
manifest. `submit_pipeline.sh` chains prepare/train/evaluate with dependencies.
The shared manifest contains all three chronological splits.

Training requests two nodes, four GPU tasks/node, six CPUs/task, and 12 hours.
It uses core 8, halo 2 (12×12 input / 72×72 output), 8 patches/day, 4 validation
patches/day, batch size 4/process, width 96, six groups of six blocks, six heads.

By default training initializes from scratch. `MV_RESUME=/path/last.pt` resumes
full state; check optimizer, epoch and model configuration first. For a fresh
experiment with only a compatible encoder initialization, use the Python CLI's
`--init-backbone checkpoints/refine_6x.pt` with a new run directory. This resets
the decoders and is not full-model fine-tuning. Full-model continuation uses
`--resume` and an epoch limit above the saved epoch.

Modify `refine_downscaling/model.py` for architecture experiments, preserve the
released checkpoint, and verify 6× output geometry and checkpoint compatibility.
Keep 1990 held out; compare new results against the frozen reference metrics.
