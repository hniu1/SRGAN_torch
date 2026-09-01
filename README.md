# Joint Multivariable ClimateSwin Downscaling

This branch trains one terrain-aware transformer to downscale daily `tmin`,
`tmax`, and precipitation together:

```text
Daymet/ERA5 or GCM at 1 degree
       -> Stage-1 ClimateSwin (4x) -> 0.25 degree
       -> Stage-2 ClimateSwin (6x) -> 1/24 degree
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

## Stage 2: 0.25 degree to 1/24 degree

Stage 2 keeps the variable-aware SwinV2 representation backbone, but does not
reuse the 4x reconstruction head unchanged. It uses a 2x PixelShuffle stage
followed by a 3x stage, fusing native 1/24-degree terrain at both scales. The
default patch is deliberately smaller because a 12x12 LR context reconstructs
a 72x72 HR field; the central 8x8 LR / 48x48 HR region is supervised.

| Step | Entry point | Frontier launcher |
|---:|---|---|
| 5 | `pipeline_05_prepare_stage2.py` | `slurm/05_prepare_stage2.slurm` |
| 6 | `pipeline_06_train_stage2.py` | `slurm/06_train_stage2.slurm` |
| 7 | `pipeline_07_evaluate_stage2.py` | `slurm/07_evaluate_stage2.slurm` |
| 8 | `pipeline_08_downscale_stage2.py` | `slurm/08_downscale_stage2.slurm` |

Submit preparation, training, and evaluation in dependency order with:

```bash
bash submit_stage2_pipeline.sh
```

The preparation step validates all paired files but does not copy the roughly
200 GB uncompressed daily archive. It writes a lightweight manifest, time
indexes, coordinates, and LR/HR terrain arrays. Each worker opens the three
variables independently and reads only the aligned LR and HR patch requested
for that sample. Missing values are masked, finite negative precipitation is
clamped to zero, and Kelvin temperature sources are converted to Celsius on
read.

The input is `*_0p25deg.nc` and the fine-resolution truth is `*_trim.nc`. The
archive's `*_0p25degto0p0416deg.nc` files are pre-interpolated coarse fields;
they are useful as a baseline, but are deliberately not used as training
targets.

The Stage-2 launcher warm-starts the compatible variable stem, seasonal/static
fusion, and Swin encoder from `artifacts/runs/climateswin_v1/best.pt`. The 6x
upsampling stages and variable decoders start fresh; use `MV_STAGE1_CHECKPOINT`
to choose another source checkpoint. Use `MV_STAGE2_RESUME` to resume a Stage-2
checkpoint instead. Stage-2 paths can be overridden with `MV_STAGE2_DATA_DIR`
and `MV_STAGE2_RUN_DIR`.

Initial training uses observed paired 0.25-degree fields as inputs and
1/24-degree fields as targets, which measures the isolated second-stage skill.
For the strongest final 1-degree-to-1/24-degree cascade, a later fine-tuning
round should mix in Stage-1-generated 0.25-degree inputs to reduce the small
train/inference distribution shift.

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
4. Two PixelShuffle stages reconstruct the grid: 2x + 2x for Stage 1, or 2x +
   3x for Stage 2.
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
