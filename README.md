# SRGAN Patch-Training Pipeline

This repository trains and evaluates the Stage-1 temperature downscaler:

```text
Daymet/ERA5 or GCM 1° -> SRGAN -> 0.25°
```

The active pipeline uses 8×8 LR patches, reflection padding, PixelShuffle,
generator pretraining, and a ten-year 1980–1989 training dataset. The numbered
filenames show the required execution order.

## Active Stage-1 Pipeline

| Step | Purpose | Python entry point | Frontier job |
|---:|---|---|---|
| 1 | Prepare 1980–1989 paired Daymet data | `pipeline_01_prepare_daymet.py` | `pipeline_01_prepare_tmax_10yr.slurm` |
| 2 | Train the 8×8 patch model | `pipeline_02_train_stage1_patch.py` | `pipeline_02_train_stage1_patch.slurm` |
| 3 | Evaluate on unseen paired 1990 Daymet | `pipeline_03_evaluate_daymet_1990.py` | `pipeline_03_evaluate_daymet_1990.slurm` |
| 4 | Diagnose behavior on GCM input | `pipeline_04_diagnose_gcm.py` | `pipeline_04_diagnose_gcm.slurm` |

Run the complete workflow in order:

```bash
sbatch pipeline_01_prepare_tmax_10yr.slurm
sbatch pipeline_02_train_stage1_patch.slurm
sbatch pipeline_03_evaluate_daymet_1990.slurm
sbatch pipeline_04_diagnose_gcm.slurm
```

Use Slurm dependencies when submitting the workflow from scratch so downstream
steps run only after upstream success.

## Current Versions

```text
Prepared data: tmax_stage1_patch_10yr_data
Model:         tmax_stage1_patch_pixelshuffle_10yr
Training:      1980–1989 (3,653 days; random 80/20 split)
External test: 1990
LR patch:      8×8
HR patch:      32×32
Scale:         4×
```

### Experimental terrain-aware version

The following separate workflow preserves the current model while testing a
deeper generator conditioned on native 0.25° elevation:

```bash
python pipeline_01b_prepare_hr_elevation.py
sbatch pipeline_02_train_stage1_hr_elev.slurm
sbatch pipeline_03_evaluate_daymet_1990_hr_elev.slurm
```

The implementation is isolated in `pipeline_02_train_stage1_hr_elev.py`.
The existing `pipeline_02_train_stage1_patch.py` and its current-model launcher
remain unchanged by the terrain-aware experiment.

```text
Model version: tmax_stage1_patch_hr_elev_deep_10yr
Training patch: 16×16 LR -> 64×64 HR
LR trunk:       4 residual blocks, 96 channels
HR fusion:      2 residual blocks
HR predictors:  elevation, upscaled LR elevation, elevation anomaly
```

The new output folder reuses the prepared ten-year arrays through symbolic
links and contains both LR and HR elevation fields. This avoids duplicating
1.8 GB or rerunning data preparation; the original
`tmax_stage1_patch_pixelshuffle_10yr` checkpoint is unchanged.

Prepared arrays are stored under `output/<data-version>/`. Scalers and model
checkpoints are stored under `models/<version>/`.

## Shared Model Modules

These are libraries imported by pipeline entry points and are not submitted
directly:

- `srgan_torch.py` — generator and discriminator definitions.
- `patch_dataset.py` — lazy paired LR/HR patch sampling.
- `loss_torch.py` — content and adversarial loss wrappers.
- `dataread_mem.py` — streaming preparation for large multi-year arrays.
- `dataread.py` — older in-memory data utilities still used by legacy code.
- `scalers.py` — serializable custom scalers.

## Evaluation Guidance

Use Step 3 for quantitative daily accuracy because its 1° input and 0.25°
truth are paired observations from the same dates. It reports bias, MAE, RMSE,
temporal means, and improvement over bilinear interpolation.

Do not interpret a historical GCM day minus the same calendar Daymet day as a
daily forecast error. A free-running climate model is not synchronized with
observed weather. Step 4 is intended for climatologies, distributions, seasonal
cycles, and extremes.

## Stage 2 and Legacy Scripts

- `stage2_train_patch.py` — experimental 0.25° -> 0.0416° Stage-2 trainer.
- `legacy_prepare_stage2_tmin_frontier.slurm` — older Frontier tmin preparation.
- `legacy_prepare_stage2_tmin_andes.slurm` — older Andes tmin preparation.
- `legacy_downscale_gcm_two_stage.py` and `.slurm` — older two-stage GCM workflow.

These files are retained for reference but are not part of the validated,
numbered Stage-1 pipeline.

## Utilities

- `utility_plot_training_losses.py`
- `utility_postprocess_predictions.py`
- `utility_inverse_scale_gcm.py`

## Temporary Compatibility File

`daymet_stage1_evaluate.py` is a small forwarding entry point retained only for
already queued Slurm job `5167356`. New runs must use
`pipeline_03_evaluate_daymet_1990.py` or its `.slurm` launcher. The compatibility
file can be removed after that queued job finishes.
