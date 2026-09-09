# Evaluation

Follow the evaluation commands in the repository README. Use
`pipeline_03_evaluate.py --data-dir daymet/prepared --checkpoint checkpoints/refine_6x.pt`
with a fresh `--output-dir`, `--split test`, `--batch-size 1`, `--amp` and
`--enforce-temperature-order`. `--max-days 1` provides a short experiment;
omitting it evaluates all 365 days. `--no-save-predictions` reduces disk usage.

`slurm/03_evaluate.slurm` evaluates the full year without predictions.
`slurm/05_save_predictions.slurm` saves full-year predictions for plotting;
`slurm/06_plot_spatial_statistics.slurm` produces spatial diagnostics.
The full year can require about 18.5 GB for prediction arrays.

Plot with `utility_plot_spatial_statistics.py --data-dir daymet/prepared
--evaluation-dir YOUR_EVALUATION_DIR --split test --tile-rows 21`.
Partial runs must be labeled with their actual day count. Do not label one-day
percentiles as annual statistics. Historical results are under docs/reference_1990.
