# REFINE Downscaling

**REFINE** (Resolution-Enhancement Framework Integrating Artificial Intelligence
for Natural and Energy Systems) is a terrain-aware Swin Transformer framework
for downscaling environmental fields. It jointly predicts daily minimum
temperature (`tmin`), maximum temperature (`tmax`), and precipitation (`prcp`),
using elevation and seasonal information to reconstruct finer spatial detail.

The current workflow supports **approximately 25 km → 5 km → 1 km** downscaling
through two independently trained 5× models:

| Stage | Input grid | Output grid | Approximate resolution |
| --- | --- | --- | --- |
| 1 | 0.25° (`025deg`) | 0.05° (`005deg`) | 25 → 5 km |
| 2 | 0.05° (`005deg`) | 0.01° (`001deg`) | 5 → 1 km |

Each stage increases both spatial dimensions by five. Degree spacing is exact;
kilometer labels are approximate and vary with latitude. Both models use the
same architecture, with separate learned weights and checkpoints. Stage 1
predictions can be passed directly into Stage 2 for a total 25× increase in
spatial resolution along each axis.

## Installation

Clone the repository and run the commands below from its root directory.
The reference environment uses Python 3.12 and PyTorch 2.8.0 with ROCm 6.4.
A compatible GPU is recommended for training; the pipeline also supports CPU
execution for small tests.

Create and activate a Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

For the reference AMD ROCm environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins an AMD ROCm build of PyTorch. For CPU or NVIDIA CUDA
systems, install the appropriate PyTorch build for your hardware, then install
the remaining dependencies:

```bash
python -m pip install numpy==2.4.2 netCDF4==1.7.4 cftime==1.6.5 matplotlib==3.10.8
```

The repository contains source code and tests. Training data, DEM files, and
trained checkpoints must be supplied separately.

## User guide

### 1. Organize the training data

The current data reader expects aligned annual Daymet NetCDF files in this
layout, repeated for `tmin`, `tmax`, and `prcp`:

```text
data/daymet_TVA_latlon/
├── 025deg/
│   ├── tmin/daymet_TVA_tmin_1982_025deg.nc
│   ├── tmax/daymet_TVA_tmax_1982_025deg.nc
│   └── prcp/daymet_TVA_prcp_1982_025deg.nc
├── 005deg/
│   ├── tmin/daymet_TVA_tmin_1982_005deg.nc
│   ├── tmax/daymet_TVA_tmax_1982_005deg.nc
│   └── prcp/daymet_TVA_prcp_1982_005deg.nc
└── 001deg/
    ├── tmin/daymet_TVA_tmin_1982_001deg.nc
    ├── tmax/daymet_TVA_tmax_1982_001deg.nc
    └── prcp/daymet_TVA_prcp_1982_001deg.nc
```

Each file must contain its named variable with `time,lat,lon` dimensions,
one-dimensional latitude/longitude coordinates, and CF time coordinates.
Supported temperature units include Celsius and Kelvin; training converts
Kelvin to Celsius. Precipitation must be in daily millimeters.

The grids must share a geographic extent, with five fine cells spanning each
coarse cell in both dimensions. The TVA grids have shapes 24×44, 120×220, and
600×1100. Preparation validates coordinate alignment and matching timestamps
across variables and resolutions. Input coordinates must increase with latitude
and longitude. The pipeline does not perform regridding.

Supply a DEM at each resolution, already aligned with the climate grid. Each
DEM NetCDF must contain matching `lat` and `lon` coordinates and an elevation
variable named `DEM`, `dem`, `elevation`, or `elev`, with elevation in meters and
spatial dimensions ordered `lat,lon`.

### 2. Prepare both stages

Set the locations of your data and terrain files:

```bash
export DATA_ROOT=/path/to/daymet_TVA_latlon
export DEM_025=/path/to/dem_025deg.nc
export DEM_005=/path/to/dem_005deg.nc
export DEM_001=/path/to/dem_001deg.nc

python pipeline_01_prepare.py --stage 1 --data-root "$DATA_ROOT" \
  --dem-lr "$DEM_025" --dem-hr "$DEM_005" \
  --output-dir daymet/prepared/stage1

python pipeline_01_prepare.py --stage 2 --data-root "$DATA_ROOT" \
  --dem-lr "$DEM_005" --dem-hr "$DEM_001" \
  --output-dir daymet/prepared/stage2
```

The default chronological split is:

| Split | Years |
| --- | --- |
| Training | 1982–2010 |
| Validation | 2011–2015 |
| Testing | 2016–2020 |

Change the split with `--train-start`, `--train-end`, `--val-start`, `--val-end`,
`--test-start`, and `--test-end`. End years are exclusive; apply the same split
settings to both stages. All requested annual files must be available.

Preparation creates lightweight indexes and static fields, leaving the source
climate files in place. Training reads NetCDF patches lazily. Normalization is
fitted on training inputs only: standardization for temperature and `log1p`
followed by standardization for precipitation. Missing values are masked, and
source timestamps determine seasonal conditioning. Use a new output directory
when changing an existing preparation configuration.

### 3. Train the models

Train each stage with its own prepared data and run directory:

```bash
python pipeline_02_train.py --data-dir daymet/prepared/stage1 \
  --run-dir artifacts/runs/refine_stage1_5x --amp

python pipeline_02_train.py --data-dir daymet/prepared/stage2 \
  --run-dir artifacts/runs/refine_stage2_5x --amp
```

Each run saves `best.pt`, `last.pt`, its configuration, and training history.
`--amp` enables mixed precision on a GPU. Training options include model size,
patch size, batch size, learning rate, and epoch count; inspect them with
`python pipeline_02_train.py --help`.

Use `--resume /path/to/last.pt` to continue a matching run. Use
`--init-backbone /path/to/checkpoint.pt` to initialize compatible encoder weights
while starting the reconstruction head and optimizer fresh.

### 4. Evaluate against reference data

Evaluate each stage on the held-out test years:

```bash
for stage in 1 2; do
  python pipeline_03_evaluate.py \
    --data-dir "daymet/prepared/stage${stage}" \
    --checkpoint "artifacts/runs/refine_stage${stage}_5x/best.pt" \
    --output-dir "artifacts/evaluation/stage${stage}" \
    --split test --max-days 1 --amp --enforce-temperature-order
done
```

Start with one day to verify the setup, then remove `--max-days 1` to evaluate
the full split. Results report bias, MAE, and RMSE alongside a bilinear baseline.
Predictions are saved by default; use `--no-save-predictions` for metrics only.

Generate spatial comparisons from saved predictions:

```bash
python utility_plot_spatial_statistics.py --data-dir daymet/prepared/stage2 \
  --evaluation-dir artifacts/evaluation/stage2 --split test
```

Stage 2 evaluation uses reference 5 km inputs. Final 25 → 1 km performance must
also be assessed using Stage 1 predictions as Stage 2 inputs, since errors from
the first model can propagate through the cascade.

### 5. Run 25 → 5 → 1 km inference

With both trained checkpoints available, downscale one day from the coarse grid:

```bash
python pipeline_04_infer.py --data-dir daymet/prepared/stage1 \
  --checkpoint artifacts/runs/refine_stage1_5x/best.pt \
  --input "tmin=$DATA_ROOT/025deg/tmin/daymet_TVA_tmin_2016_025deg.nc" \
  --input "tmax=$DATA_ROOT/025deg/tmax/daymet_TVA_tmax_2016_025deg.nc" \
  --input "prcp=$DATA_ROOT/025deg/prcp/daymet_TVA_prcp_2016_025deg.nc" \
  --output artifacts/inference/predicted_005deg.nc \
  --end-index 1 --amp --enforce-temperature-order
```

Pass that output to the second model:

```bash
python pipeline_04_infer.py --data-dir daymet/prepared/stage2 \
  --checkpoint artifacts/runs/refine_stage2_5x/best.pt \
  --input tmin=artifacts/inference/predicted_005deg.nc \
  --input tmax=artifacts/inference/predicted_005deg.nc \
  --input prcp=artifacts/inference/predicted_005deg.nc \
  --output artifacts/inference/predicted_001deg.nc \
  --amp --enforce-temperature-order
```

Increase `--end-index` or omit it to process more timesteps. Input files must
match the model's prepared grid and use Celsius and mm/day. Each NetCDF output
contains all predicted variables with `time,lat,lon` dimensions, geographic
coordinates, source timestamps, and missing values outside the valid domain.
A JSON file records the checkpoint, inputs, and processed interval. Stage 1
outputs are converted to physical units before Stage 2 applies its own
normalization. Existing output files are protected from overwrite.

## Running on an HPC system

The training code supports distributed execution through `torchrun` and Slurm
`srun`. Example Slurm launchers are provided in `slurm/`. These launchers contain
Frontier-specific account, module, and environment settings; adapt them to your
system before submission. `MV_STAGE=1` or `MV_STAGE=2` selects the stage, with
separate default data and run directories.

## Repository structure

| Path | Purpose |
| --- | --- |
| `refine_downscaling/` | Model, data readers, preparation, transforms, and losses |
| `pipeline_01_prepare.py` | Validate data and prepare stage indexes |
| `pipeline_02_train.py` | Train a REFINE model |
| `pipeline_03_evaluate.py` | Evaluate a checkpoint against reference fields |
| `pipeline_04_infer.py` | Downscale new inputs with a trained checkpoint |
| `utility_plot_spatial_statistics.py` | Plot spatial comparisons |
| `utility_plot_training_history.py` | Plot training history |
| `slurm/` | Example HPC launchers |
| `tests/` | Model and pipeline checks |

## Verification

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Tests cover both 5× stages, training and evaluation, chained inference,
normalization, missing values, dates, grid alignment, and checkpoint consistency.
The code retains support for the earlier 6× configuration; its checkpoints
require matching 6× prepared data and cannot directly run the 5× stages.

## Authors and data reference

**Authors:** Haoran Niu and Deeksha Rastogi.

This workflow uses [Daymet](https://daymet.ornl.gov/) daily climate fields on
aligned latitude/longitude grids. The 0.01° grid is approximately 1 km; the 0.05°
and 0.25° grids provide the coarser training levels.

Thornton, P.E., Shrestha, R., Thornton, M. et al. (2021).
[Gridded daily weather data for North America with comprehensive uncertainty
quantification](https://doi.org/10.1038/s41597-021-00973-0).
*Scientific Data*, 8, 190.
