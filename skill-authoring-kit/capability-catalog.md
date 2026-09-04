# Capability catalog

Keep these capabilities separate even if one coordinating agent invokes several of them.

## `climate-downscaling`

Use when a user wants to validate climate inputs or downscale `tmin`, `tmax`, and `prcp` through Stage 1, Stage 2, or the cascade.

Inputs:

- route: `stage1`, `stage2`, or `cascade`
- one file mapping for each canonical variable
- output path and format
- start date and optional half-open time-index interval
- optional reviewed checkpoint override

Outputs:

- preflight report
- reproducible command/run record
- multivariable downscaled array or NetCDF
- warnings and provenance

Do not use it to train models, submit Slurm jobs, or characterize accuracy.

## `climate-downscaling-evaluation`

Use when a user asks how well a checkpoint performs, requests comparison with Daymet/bilinear interpolation, or wants standardized spatial statistics.

Inputs:

- stage, manifest, checkpoint, split, and optional maximum days
- whether to retain prediction arrays
- requested metrics/statistics/plots

Outputs:

- per-variable bias, MAE, RMSE, baseline MAE, and improvement
- physical-constraint diagnostics
- artifact paths and scientific limitations

Do not present single-stage held-out results as end-to-end cascade evaluation.

## `climate-downscaling-ops`

Use for Daymet preparation, fresh/resumed training, Slurm submission/status/cancellation, log diagnosis, and checkpoint recovery.

Inputs:

- requested action and stage
- exact data/run/checkpoint paths
- resource or environment overrides
- explicit approval for state-changing actions

Outputs:

- effective command and configuration
- job IDs/dependencies/states
- log and checkpoint paths
- diagnosis and next verification step

Reading scheduler state and logs is safe by default. Submission, cancellation, overwrite, and deletion require explicit approval at execution time.

## Coordinator behavior

A higher-level agent may chain the capabilities:

```text
validate -> infer -> evaluate -> summarize
                   \
                    operations only when HPC execution is required
```

The coordinator should pass artifact identities, not conversational assumptions, between capabilities.
