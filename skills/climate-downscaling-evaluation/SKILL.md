---
name: climate-downscaling-evaluation
description: Evaluate ClimateSwin downscaling against held-out Daymet data, summarize model and bilinear-baseline metrics, and generate standardized spatial-statistics plots. Use for quantitative performance reports, bias analysis, or comparison maps. Do not use for training, job submission, or downscaling new climate inputs.
---

# Climate Downscaling Evaluation

Evaluate a named checkpoint on a named data split and keep the run identity attached to every reported metric and plot.

## Workflow

1. Confirm the stage, checkpoint, manifest, split, and whether predictions must be retained.
2. Read [references/evaluation.md](references/evaluation.md) for the correct pipeline and storage implications.
3. Run evaluation locally only when resources are appropriate. Use `$climate-downscaling-ops` for scheduler submission or monitoring.
4. Summarize an existing result deterministically:

   ```bash
   python skills/climate-downscaling-evaluation/scripts/summarize_evaluation.py \
     artifacts/runs/climateswin_stage2_v1/test_1990/evaluation_summary.json
   ```

5. Read [references/metrics.md](references/metrics.md) before interpreting the statistics.
6. Read [references/spatial-plots.md](references/spatial-plots.md) before creating or reviewing comparison maps.
7. Report failures and warnings separately from scientific findings.

## Guardrails

- Never compare metrics unless the split, variables, units, masks, and aggregation are compatible.
- Label the bilinear result as a baseline, not a competing trained model.
- Report bias, MAE, RMSE, MAE improvement, sample count, and temperature-order handling together.
- Do not infer generalization beyond the evaluated years/domain.
- Full Stage 2 prediction arrays are roughly 18 GB for a 365-day float32 three-variable evaluation. Prefer `--no-save-predictions` unless spatial plots or retained predictions are explicitly required.
- Do not submit jobs, overwrite evaluation artifacts, or delete large predictions without explicit authorization.

## Output

Return a compact evaluation record with stage/checkpoint, split and days, per-variable metrics and units, baseline comparison, physical-constraint status, artifact paths, and interpretation limits.
