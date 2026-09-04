# Inference commands

Run commands from the repository root. Use the project Python environment on Frontier:

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm/bin/python pipeline_04_downscale_gcm.py --help
```

## Stage 1: 1 degree to 0.25 degree

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm/bin/python pipeline_04_downscale_gcm.py \
  --data-dir artifacts/data/daymet_mv_1980_1990 \
  --checkpoint artifacts/runs/climateswin_v1/best.pt \
  --input tmin=/path/tmin.nc \
  --input tmax=/path/tmax.nc \
  --input prcp=/path/prcp.nc \
  --output /path/downscaled_025deg.nc \
  --start-date YYYY-MM-DD \
  --format netcdf \
  --amp \
  --enforce-temperature-order
```

Use `--start-index` and `--end-index` to select a half-open time interval. Choose `--batch-size` conservatively for available GPU memory.

## Stage 2: 0.25 degree to 1/24 degree

Use `pipeline_08_downscale_stage2.py`, which exposes the same interface. Always supply the Stage 2 data directory and checkpoint because the shared parser retains Stage 1 defaults:

```bash
/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm/bin/python pipeline_08_downscale_stage2.py \
  --data-dir artifacts/data/daymet_mv_stage2_1980_1990 \
  --checkpoint artifacts/runs/climateswin_stage2_v1/best.pt \
  --input tmin=/path/tmin_025deg.nc \
  --input tmax=/path/tmax_025deg.nc \
  --input prcp=/path/prcp_025deg.nc \
  --output /path/downscaled_1over24deg.nc \
  --start-date YYYY-MM-DD \
  --format netcdf \
  --amp \
  --enforce-temperature-order
```

## Cascade: 1 degree to 1/24 degree

Run Stage 1 first. Its multivariable NetCDF output can be passed as the path for all three Stage 2 inputs:

```bash
python pipeline_08_downscale_stage2.py \
  --input tmin=/path/downscaled_025deg.nc \
  --input tmax=/path/downscaled_025deg.nc \
  --input prcp=/path/downscaled_025deg.nc \
  --output /path/downscaled_1over24deg.nc \
  --start-date YYYY-MM-DD \
  --format netcdf \
  --amp \
  --enforce-temperature-order
```

The current writer uses index dimensions `y` and `x`. Retain the source grid/georeferencing metadata alongside the result; do not assume the output NetCDF alone fully describes its geographic coordinates.
