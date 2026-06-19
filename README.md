# SRGAN North America Downscaling

This folder contains the PyTorch SRGAN workflow used to train and apply
North America downscaling models. The main daily workflow is trained over the
full North America domain in two resolution stages:

1. Stage 1: 1 degree -> 1/4 degree
2. Stage 2: 1/4 degree -> 1/24 degree, stored in the code as 0.0416 degree

The trained stages are then chained for GCM downscaling so coarse climate model
fields can be converted to daily high-resolution North America fields.

## Current Branch And Git Notes

This work is organized on the branch:

```bash
SRGAN_Batch_North_America
```

The intended GitHub remote is:

```bash
https://github.com/hniu1/SRGAN_torch.git
```

On this Lustre workspace, the directory has a special read-only `.git` entry.
Because of that, normal Git discovery would otherwise climb to the parent
`cli138` repository. A local Git metadata directory named `.srgan_git` is used
for this project, and `~/.bashrc` routes `git` commands in this folder to that
metadata.

If Git ever appears to show the parent `cli138` repo again, use the explicit
form:

```bash
git --git-dir=.srgan_git --work-tree=. status
git --git-dir=.srgan_git --work-tree=. remote -v
```

## Resolution And Version Map

The daily model versions are organized by variable and resolution stage.

| Variable | Stage | Version | Resolution |
| --- | --- | --- | --- |
| prcp / pr | Stage 1 | `dy_v0.3` | 1 degree -> 0.25 degree |
| prcp / pr | Stage 2 | `dy_v0.4` | 0.25 degree -> 0.0416 degree |
| tmax | Stage 1 | `dy_v0.5` | 1 degree -> 0.25 degree |
| tmax | Stage 2 | `dy_v0.6` | 0.25 degree -> 0.0416 degree |
| tmin | Stage 1 | `dy_v0.7` | 1 degree -> 0.25 degree |
| tmin | Stage 2 | `dy_v0.8` | 0.25 degree -> 0.0416 degree |

Older 3-hour experiments are stored as `3h_*` versions:

| Version | Notes |
| --- | --- |
| `3h_v0.1` | 2015-2019 temperature, trained on Frontier |
| `3h_v0.2` | 2015-2019 temperature |
| `3h_v0.3_prcp` | 2015-2019 precipitation |

## Main Workflow

The high-level workflow is:

1. Prepare cached Daymet/ERA5 training arrays.
2. Train Stage 1 SRGAN for 1 degree -> 0.25 degree.
3. Train Stage 2 SRGAN for 0.25 degree -> 0.0416 degree.
4. Apply Stage 1 and Stage 2 models to GCM data.
5. Save daily yearly NetCDF files on the final 0.0416 degree grid.

The training data covers the North America domain. Daily training versions use
the 1980-2020 period according to `models/versions.txt`.

## Important Files

### Model Definition

| File | Purpose |
| --- | --- |
| `srgan_torch.py` | Generator and discriminator architectures, including the SRGAN variants used for low-resolution and high-resolution stages. |
| `loss_torch.py` | Loss wrappers for initial generator training, adversarial generator training, and discriminator training. |
| `scalers.py` | Scaling utilities used by the data pipeline. |

### Data Preparation

| File | Purpose |
| --- | --- |
| `dataread.py` | Original Daymet/ERA5 data reader and cached array writer. |
| `dataread_mem.py` | Memory-safe reader that streams yearly NetCDF files and writes `x_train`, `x_test`, `y_train`, and `y_test` as `.npy` arrays. |
| `prepare_daymet.py` | Helper for preparing Daymet-related inputs. |
| `npy_to_yearly_netcdf.py` | Converts numpy predictions back to yearly NetCDF-style outputs. |

### Daily Training

| File | Purpose |
| --- | --- |
| `train_daily_frontier_srun.py` | Frontier-ready distributed daily SRGAN training script. |
| `train_daily_frontier_srun_2nd.py` | Daily training variant used for second-stage/high-resolution runs. |
| `train_daily_frontier_srun_test_parameter.py` | Sweep-friendly training script controlled by environment variables from the SLURM script. |
| `train_daily_frontier_srun_tmax.py` | Daily training variant for `tmax`. |
| `train_daily_frontier_srun_tmin.py` | Daily training variant for `tmin`. |
| `train_3h_prcp.py` | Older 3-hour precipitation training script. |

### SLURM Launch Scripts

| File | Purpose |
| --- | --- |
| `srgan_srun.sh` | Frontier SLURM launcher for daily training. Currently configured around `train_daily_frontier_srun_2nd.py`. |
| `srgan_srun_test_parameter.sh` | Parameter-sweep launcher. Use `sbatch --export=ALL,...` to override variable, version, epochs, batch size, and loss weights. |
| `srgan_test.sh` | Test/evaluation launcher. |
| `srun_interactive.sh` | Interactive run helper. |
| `run_temp.sh` | Small temperature-related run helper. |

### GCM Downscaling And Postprocessing

| File | Purpose |
| --- | --- |
| `gcm_downscaling.py` | Earlier GCM downscaling workflow. |
| `gcm_downscaling_dims_yearly.py` | Main yearly GCM downscaling workflow with dimension-aware NetCDF output. |
| `gcm_downscaling_dims_infer_yearly_frontier.sh` | SLURM script for yearly downscaling / save-to-NetCDF workflow. |
| `gcm_downscaling_dims_save_yearly_andes.sh` | Andes save-to-NetCDF helper. |
| `gcm_downscling_scale.py` | Scaling helper for GCM downscaling. |
| `gcm_ds_npy_to_netcdf_save_yearly.py` | Converts GCM downscaled `.npy` outputs to yearly NetCDF. |
| `gcm_ds_npy_to_netcdf_save_yearly_srun.sh` | SLURM launcher for the conversion script. |
| `postproc.py` | Postprocessing utilities. |

### Plotting And Checks

| File | Purpose |
| --- | --- |
| `plot_loss.py` | Plot training loss curves. |
| `plot_pred25_3row_compare.py` | Compare predictions visually. |
| `check_downscaled_pr.py` | Check precipitation downscaled outputs. |
| `utils/plot_utils.py` | Shared plotting helpers. |

## Data And Output Directories

These directories are intentionally not tracked in Git because they contain
large generated arrays, model checkpoints, logs, or NetCDF outputs.

| Directory | Purpose |
| --- | --- |
| `output/<version>/` | Cached train/test arrays and model predictions. Expected files include `x_train.npy`, `x_test.npy`, `y_train.npy`, `y_test.npy`, and optional elevation arrays. |
| `models/<version>/` | Checkpoints, scalers, and loss histories for each model version. |
| `logs/` | SLURM stdout/stderr logs. |
| `gcm_ds/<downscale-version>/` | GCM downscaling numpy outputs and converted yearly NetCDF products. |

Only the version marker files are kept in Git:

```bash
models/versions.txt
gcm_ds/versions.txt
```

## Typical Training Commands

Stage 1 precipitation example, 1 degree -> 0.25 degree:

```bash
sbatch --export=ALL,VAR=prcp,VERSION=dy_v0.3,INITIAL_TRAINING=1 srgan_srun_test_parameter.sh
```

Stage 2 precipitation example, 0.25 degree -> 0.0416 degree:

```bash
sbatch --export=ALL,VAR=prcp,VERSION=dy_v0.4 srgan_srun.sh
```

Temperature versions follow the same pattern:

```bash
# tmax
sbatch --export=ALL,VAR=tmax,VERSION=dy_v0.5,INITIAL_TRAINING=1 srgan_srun_test_parameter.sh
sbatch --export=ALL,VAR=tmax,VERSION=dy_v0.6 srgan_srun.sh

# tmin
sbatch --export=ALL,VAR=tmin,VERSION=dy_v0.7,INITIAL_TRAINING=1 srgan_srun_test_parameter.sh
sbatch --export=ALL,VAR=tmin,VERSION=dy_v0.8 srgan_srun.sh
```

Check the SLURM scripts before submitting because node count, batch size,
epoch count, loss weights, and model version are often edited during
experiments.

## GCM Downscaling

After both stages are trained for a variable, use the yearly GCM workflow to
chain the low-resolution and high-resolution SRGAN models.

Default version pairing in `gcm_downscaling_dims_infer_yearly_frontier.sh`:

| Variable | Stage 1 model | Stage 2 model |
| --- | --- | --- |
| `pr` | `dy_v0.3` | `dy_v0.4` |
| `tmax` | `dy_v0.5` | `dy_v0.6` |
| `tmin` | `dy_v0.7` | `dy_v0.8` |

Example:

```bash
sbatch --export=ALL,VAR=pr,SCENARIO=ssp585,DOWNSCALE_VERSION=0.3,YEAR_START=2020,YEAR_END=2060 gcm_downscaling_dims_infer_yearly_frontier.sh
```

`YEAR_END` is exclusive, so `YEAR_START=2020,YEAR_END=2060` processes
2020-2059.

## Notes For Future Cleanup

- The folder currently keeps multiple training variants that were useful during
  development. The clearest long-term entry points are the daily Frontier
  training scripts, the parameter-sweep SLURM launcher, and
  `gcm_downscaling_dims_yearly.py`.
- The generated data directories are very large and should stay out of Git.
- When adding a new variable or version, update `models/versions.txt`,
  `gcm_ds/versions.txt`, and the version pairing in the GCM launcher.
