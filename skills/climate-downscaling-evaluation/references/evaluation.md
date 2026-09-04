# Evaluation workflow

Run from the repository root with the project Python environment.

## Stage 1

```bash
python pipeline_03_evaluate_mvswin.py \
  --data-dir artifacts/data/daymet_mv_1980_1990 \
  --checkpoint artifacts/runs/climateswin_v1/best.pt \
  --output-dir artifacts/runs/climateswin_v1/test_1990 \
  --split test \
  --batch-size 1 \
  --amp \
  --no-save-predictions \
  --enforce-temperature-order
```

## Stage 2

`pipeline_07_evaluate_stage2.py` exposes the same arguments:

```bash
python pipeline_07_evaluate_stage2.py \
  --data-dir artifacts/data/daymet_mv_stage2_1980_1990 \
  --checkpoint artifacts/runs/climateswin_stage2_v1/best.pt \
  --output-dir artifacts/runs/climateswin_stage2_v1/test_1990 \
  --split test \
  --batch-size 1 \
  --amp \
  --no-save-predictions \
  --enforce-temperature-order
```

Use `--max-days N` only for smoke tests and label partial results clearly. Omit `--no-save-predictions` when spatial-statistics generation is required.

## Spatial statistics

After predictions exist in the evaluation directory:

```bash
python utility_plot_spatial_statistics.py \
  --data-dir artifacts/data/daymet_mv_stage2_1980_1990 \
  --evaluation-dir artifacts/runs/climateswin_stage2_v1/test_1990 \
  --split test \
  --tile-rows 21
```

Stage 2 scheduler workflow:

- `slurm/07_evaluate_stage2.slurm`: metrics without saved predictions.
- `slurm/09_spatial_statistics_stage2.slurm`: rerun while saving predictions.
- `slurm/10_plot_spatial_statistics.slurm`: tile-wise spatial statistics and plots.

Use the operations skill before interacting with Slurm.
