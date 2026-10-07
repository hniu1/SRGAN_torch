# Genesis TVA Daymet downscaling

REFINE runs two independently trained stages: `025deg → 005deg` and
`005deg → 001deg`. Each stage expands both spatial dimensions by 5. The
architecture, objectives and optimizer follow the existing supervised pipeline;
Stage 2 initially trains against reference 5 km inputs. Training it on Stage 1
predictions or jointly fine-tuning the cascade would be a separate experiment.

## Prepare aligned data

The source layout is:

```text
daymet_TVA_latlon/
  025deg/tmin/daymet_TVA_tmin_1982_025deg.nc
  005deg/tmin/daymet_TVA_tmin_1982_005deg.nc
  001deg/tmin/daymet_TVA_tmin_1982_001deg.nc
  ... same layout for tmax, prcp and years through 2020
```

Files must contain `tmin`, `tmax` or `prcp` with `time,lat,lon` dimensions,
one-dimensional lat/lon coordinates and CF time. Preparation verifies units,
timestamps, grid spacing and the alignment of coarse cells with groups of five
fine cells. It stores file indexes separately from calendar day of year, so
missing dates do not shift samples. Daymet has 365 records per year, including
leap years; the actual file timestamps determine seasonal conditioning.

Supply one DEM at each resolution. DEM files must have matching `lat` and `lon`
coordinates, spatial dimensions ordered `lat,lon`, and a variable named `DEM`,
`dem`, `elevation` or `elev`, with elevation in meters. They must already be on
the Daymet grids: the pipeline does not regrid terrain or substitute zero terrain.

Run from the repository root after activating the Frontier environment:

```bash
source scripts/frontier_env.sh

# Replace these paths with the aligned terrain files.
export DEM_025=/path/to/dem_025deg.nc
export DEM_005=/path/to/dem_005deg.nc
export DEM_001=/path/to/dem_001deg.nc

python pipeline_01_prepare.py --stage 1 \
  --dem-lr "$DEM_025" --dem-hr "$DEM_005"
python pipeline_01_prepare.py --stage 2 \
  --dem-lr "$DEM_005" --dem-hr "$DEM_001"
```

Use allocated compute resources for preparation over many years. Default splits
are training 1982–2010, validation 2011–2015 and testing 2016–2020. End years are
exclusive. For a small initial index, pass these flags to **both** commands:

```bash
--train-start 1982 --train-end 1988 \
--val-start 1988 --val-end 1990 \
--test-start 1990 --test-end 1991
```

`--data-root` overrides the shared source directory. `--output-dir` overrides
`daymet/prepared/stageN`. Annual climate arrays remain in the read-only source
directory; only terrain, masks, coordinates, time indexes and a manifest are
written locally. Existing manifests are protected from overwrite; use a new
output directory for another split or terrain configuration.

Normalization uses all finite training LR values for each stage, streaming eight
days at a time. Temperature retains the existing standard transform;
precipitation retains `log1p` followed by standardization. LR and HR share a
stage's transform. Validation/test values never fit those statistics. Missing
climate values are zero-filled after normalization and excluded through masks;
negative precipitation is clamped to zero on training reads. Static domain masks
use finite DEM and climate values on the first training timestep.

## Train and evaluate separate checkpoints

Inside an allocated GPU session:

```bash
python pipeline_02_train.py --data-dir daymet/prepared/stage1 \
  --run-dir artifacts/runs/refine_stage1_5x --amp
python pipeline_02_train.py --data-dir daymet/prepared/stage2 \
  --run-dir artifacts/runs/refine_stage2_5x --amp

python pipeline_03_evaluate.py --data-dir daymet/prepared/stage1 \
  --checkpoint artifacts/runs/refine_stage1_5x/best.pt \
  --output-dir artifacts/evaluation/stage1 --max-days 1 --amp
python pipeline_03_evaluate.py --data-dir daymet/prepared/stage2 \
  --checkpoint artifacts/runs/refine_stage2_5x/best.pt \
  --output-dir artifacts/evaluation/stage2 --max-days 1 --amp
```

Each stage saves `best.pt`, `last.pt`, training history and its configuration.
The scale is taken from the prepared manifest. Stage/grid/normalization checks
prevent loading a stage checkpoint with the other stage's prepared data, even
though both use a 5× architecture. `--resume` resumes that stage's training;
`--init-backbone` optionally imports compatible encoder weights while starting
the reconstruction head and optimizer fresh. A legacy 6× checkpoint can supply
compatible encoder weights, but its 6× head cannot perform 5× inference.

For Slurm after both indexes are prepared:

```bash
mkdir -p logs
MV_STAGE=1 sbatch slurm/02_train.slurm
MV_STAGE=2 sbatch slurm/02_train.slurm
```

The launchers retain the existing node/GPU allocation and training settings.
For preparation → training → evaluation of a single stage, submit:

```bash
MV_STAGE=1 MV_DEM_LR="$DEM_025" MV_DEM_HR="$DEM_005" bash submit_pipeline.sh
MV_STAGE=2 MV_DEM_LR="$DEM_005" MV_DEM_HR="$DEM_001" bash submit_pipeline.sh
```

Use this submission workflow **instead of** preparing those same indexes
manually. Preparation refuses an existing manifest. Launchers accept
`MV_SOURCE_ROOT`, `MV_DATA_DIR`, `MV_RUN_DIR`, `MV_CHECKPOINT`, and
`MV_TRAIN_START/END`, `MV_VAL_START/END`, `MV_TEST_START/END`. The split overrides
apply to preparation. Clear stage-specific overrides when switching stages.

## Chain 25 → 5 → 1 km inference

Start with one day, then increase `--end-index`. The source CF time determines
dates; `--start-date` is optional and, when provided, must match input index zero.

```bash
DATA_ROOT=/lustre/orion/proj-shared/cli138/dr6/Daymet1km/daymet_TVA_latlon

python pipeline_04_infer.py --data-dir daymet/prepared/stage1 \
  --checkpoint artifacts/runs/refine_stage1_5x/best.pt \
  --input "tmin=$DATA_ROOT/025deg/tmin/daymet_TVA_tmin_2016_025deg.nc" \
  --input "tmax=$DATA_ROOT/025deg/tmax/daymet_TVA_tmax_2016_025deg.nc" \
  --input "prcp=$DATA_ROOT/025deg/prcp/daymet_TVA_prcp_2016_025deg.nc" \
  --output artifacts/inference/predicted_005deg.nc \
  --end-index 1 --amp --enforce-temperature-order

python pipeline_04_infer.py --data-dir daymet/prepared/stage2 \
  --checkpoint artifacts/runs/refine_stage2_5x/best.pt \
  --input tmin=artifacts/inference/predicted_005deg.nc \
  --input tmax=artifacts/inference/predicted_005deg.nc \
  --input prcp=artifacts/inference/predicted_005deg.nc \
  --output artifacts/inference/predicted_001deg.nc \
  --amp --enforce-temperature-order
```

The output includes `time,lat,lon`, Celsius/mm/day variables, missing-value masks
and a JSON run record. Stage 1 predictions are converted back to physical units
before Stage 2 applies its own normalization. Output files are protected from
overwrite. `slurm/04_infer.slurm` uses reference inputs for the selected stage;
to chain it, set all three `MV_*_INPUT` paths to the Stage 1 output.

Stage 2 evaluation above measures performance with reference 5 km inputs.
Assess final cascade performance separately against the corresponding 1 km
targets; it also includes Stage 1 errors.

To plot saved single-stage evaluation predictions:

```bash
python utility_plot_spatial_statistics.py --data-dir daymet/prepared/stage2 \
  --evaluation-dir artifacts/evaluation/stage2 --split test
```

## Verification

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Tests cover both 5× stages and backward passes, train-only normalization,
missing cells, nonconsecutive dates, geographic alignment, checkpoint mismatches,
and chained inference. They also retain the released 6× demo regression tests.
