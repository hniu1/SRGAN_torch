# Preparation and training: 1/4° to 1/24°

Training is optional and separate from the inference demo. Copy paired
`Daymet_ERA5_{tmin,tmax,prcp}_dy_YEAR_{0p25deg,trim}.nc` for 1980–1989 from the
source directory listed in `daymet/README.md` into `daymet/data/` before preparing
training. The 1990 pairs and both DEMs are already bundled. This adds roughly
200 GB of uncompressed source data; verify available space before copying.

```bash
python pipeline_01_prepare.py --data-root daymet/data --dem-root daymet/dem \
  --normalization-manifest daymet/normalization.json \
  --output-dir artifacts/data/daymet_training
```

Splits are chronological: training 1980–1987, validation 1988–1989, test 1990.
Year-end arguments are exclusive. Preparation reuses frozen training-only
normalization for compatibility with the pretrained checkpoint. New variables
or a different climate/domain require explicitly fitting appropriate transforms
on training data; this preparation command does not fit new normalization.

After authorization, submit `slurm/02_train.slurm` with the prepared training
manifest. `submit_pipeline.sh` chains prepare/train/evaluate with dependencies.
The test-only demo manifest cannot be used for training.

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
