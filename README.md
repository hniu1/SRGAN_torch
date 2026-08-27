# Joint Multivariable ClimateSwin Downscaling

This branch trains one terrain-aware transformer to downscale daily `tmin`,
`tmax`, and precipitation together:

```text
Daymet/ERA5 or GCM at 1 degree
              -> ClimateSwin ->
joint fields at 0.25 degree
```

ClimateSwin uses a shared variable-aware SwinV2 encoder, shifted-window
attention on the low-resolution grid, PixelShuffle reconstruction, native
high-resolution terrain fusion, and a separate decoder for every output
variable. It is deterministic and does not use an adversarial discriminator.

## Active pipeline

| Step | Entry point | Frontier launcher |
|---:|---|---|
| 1 | `pipeline_01_prepare_multivariable.py` | `slurm/01_prepare_multivariable.slurm` |
| 2 | `pipeline_02_train_mvswin.py` | `slurm/02_train_mvswin.slurm` |
| 3 | `pipeline_03_evaluate_mvswin.py` | `slurm/03_evaluate_mvswin.slurm` |
| 4 | `pipeline_04_downscale_gcm.py` | `slurm/04_downscale_gcm.slurm` |

Submit preparation, training, and independent evaluation with dependencies:

```bash
bash submit_pipeline.sh
```

Or submit and inspect each stage separately:

```bash
sbatch slurm/01_prepare_multivariable.slurm
sbatch slurm/02_train_mvswin.slurm
sbatch slurm/03_evaluate_mvswin.slurm
```

The launchers accept these environment overrides:

```text
MV_BASE_DIR   repository path; defaults to the submission directory
MV_DATA_DIR   prepared dataset directory
MV_RUN_DIR    checkpoints, history, predictions, and metrics
```

Step 4 is intentionally not included in `submit_pipeline.sh`: it produces a
multi-decade GCM dataset and should be submitted only after the independent
test metrics have been reviewed. Its launcher currently targets the aligned
ACCESS-CM2 1980-2019 files and can be redirected with `MV_GCM_DIR` and
`MV_GCM_OUTPUT`. The Python entry point accepts arbitrary aligned inputs through
repeated `--input VARIABLE=PATH` arguments and writes NetCDF4 or NPY output.

## Chronological data split

The default preparation deliberately avoids random day splitting:

| Split | Years | Purpose |
|---|---|---|
| Training | 1980-1987 | Parameter optimization and normalization statistics |
| Validation | 1988-1989 | Checkpoint selection and early stopping |
| Test | 1990 | Independent final evaluation |

Arrays remain in physical units on disk and are memory mapped during training.
Temperature uses standard scaling. Precipitation uses `log1p` followed by
standard scaling. Transformation parameters are fitted only on training years
and stored in `manifest.json` rather than a Python pickle.

Preparation validates units from each NetCDF variable. The current Daymet
temperature files declare Celsius and are retained as Celsius; a future source
declaring Kelvin is converted to Celsius exactly once. Masked and non-finite
values are imputed on disk and excluded through the valid-data mask. Finite
negative precipitation is clamped to zero before `log1p`. Counts of missing
values and precipitation corrections are recorded per variable under
`data_quality` in the manifest.

Each variable is stored independently, while time, terrain, and coordinates are
shared:

```text
daymet_mv_1980_1990/
├── manifest.json
├── shared/{time_*.npy,elevation_*.npy,coordinates_*.npy}
└── variables/
    ├── tmin/{lr_*.npy,hr_*.npy,valid_*.npy,complete.json}
    ├── tmax/{lr_*.npy,hr_*.npy,valid_*.npy,complete.json}
    └── prcp/{lr_*.npy,hr_*.npy,valid_*.npy,complete.json}
```

The patch dataset opens these files once as read-only memory maps. For each
sample it reads the same day and spatial crop from each selected variable, then
stacks only those small crops in memory. Adding a prepared variable therefore
does not rewrite tmin, tmax, precipitation, or the shared arrays. For example:

```bash
python pipeline_01_prepare_multivariable.py \
  --output-dir artifacts/data/daymet_mv_1980_1990 \
  --variables humidity
```

Training can select any prepared subset and its channel order with
`--variables`. Repeating preparation skips complete variables by default;
`--overwrite` rebuilds only variables explicitly named by `--variables`.

The verified source locations used by the default launcher are:

```text
/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data
/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM/final-elev
```

Prepared data are written below `artifacts/data/`; run products are written
below `artifacts/runs/`. Both paths are ignored by Git.

## Patch geometry

The default training sample is:

```text
24x24 LR context -> 96x96 HR prediction
   central 16x16 -> central 64x64 supervised core
```

The four-cell LR halo supplies real surrounding context. Loss outside the
central core is masked, preventing patch boundaries from becoming a learned
signal. Random patches can include domain boundaries using reflection padding.
Validation uses deterministic, spatially distributed origins.

Full-domain inference is supported because SwinV2 uses window-relative rather
than full-image absolute positions. The 57x129 LR domain is internally padded
to attention-window multiples and the padding is removed before reconstruction.

## Model structure

1. Each dynamic variable has its own convolutional tokenizer and learned
   variable identity.
2. Local cross-variable attention creates a shared atmospheric feature at each
   LR grid cell.
3. Residual SwinV2 groups learn spatial relationships using 8x8 shifted
   windows.
4. Two PixelShuffle stages reconstruct the 4x grid.
5. HR elevation, elevation anomaly, coordinates, and the valid-data mask are
   fused during reconstruction.
6. Variable-specific decoders predict corrections to bilinearly upscaled input
   fields. The correction layers start at zero.

The default Frontier launcher uses six residual Swin groups with six blocks per
group, 96 features, and six attention heads. Channel dropout makes the encoder
less dependent on every input variable being present.

## Objectives

The primary objective combines normalized smooth-L1, MAE, and spatial-gradient
losses for every variable. It also includes:

- a `tmin <= tmax` consistency penalty;
- a coarse-grid precipitation conservation penalty; and
- central-core and valid-data masks.

Evaluation reports bias, MAE, and RMSE in physical units for every variable,
along with improvement over bilinear interpolation and the temperature-order
violation rate. Predictions and a JSON summary are saved in the run directory.

## Local verification

Use the project environment because the system Python does not contain PyTorch
or netCDF4:

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm/bin/python -m unittest discover -s tests -v
```

## Legacy baseline

The previous SRGAN source, launchers, documentation, utilities, generated
checkpoints, arrays, and logs are preserved under
[`archive/srgan_v2`](archive/srgan_v2/ARCHIVE_NOTICE.md). The active repository
root now contains only the ClimateSwin pipeline.
