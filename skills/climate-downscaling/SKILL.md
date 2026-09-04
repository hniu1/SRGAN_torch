---
name: climate-downscaling
description: Run or prepare ClimateSwin multivariable inference for tmin, tmax, and precipitation at 1 degree to 0.25 degree, 0.25 degree to 1/24 degree, or as a two-stage cascade. Use for input validation, checkpoint selection, inference commands, and output interpretation. Do not use for model training, scheduler operations, or quantitative evaluation.
---

# Climate Downscaling

Use the repository's trained ClimateSwin models to downscale aligned daily climate fields. Treat the three variables as one model input; accept separate source files so new variables remain composable.

## Workflow

1. Locate the repository root. It contains `pipeline_04_downscale_gcm.py`.
2. Identify the requested route:
   - Stage 1: 1 degree to 0.25 degree (4x)
   - Stage 2: 0.25 degree to 1/24 degree (6x)
   - Cascade: Stage 1 followed by Stage 2 (24x total)
3. Read [references/data-contract.md](references/data-contract.md) and validate every input before inference:

   ```bash
   python skills/climate-downscaling/scripts/validate_inputs.py \
     --stage stage1 \
     --input tmin=/path/tmin.nc \
     --input tmax=/path/tmax.nc \
     --input prcp=/path/prcp.nc \
     --strict \
     --full-scan
   ```

4. Read [references/inference.md](references/inference.md) and construct the command for the selected route.
5. Read [references/checkpoints.md](references/checkpoints.md) before overriding a checkpoint or data manifest.
6. State the expected output path, time range, stage, and approximate output size before launching a long inference.
7. Inspect the resulting NetCDF/NumPy metadata and report output shape, variables, time coverage, and any validation warnings.

## Guardrails

- Require all three variables: `tmin`, `tmax`, and `prcp`.
- Require temperature in degrees Celsius and precipitation in millimetres per day. Inference does not convert Kelvin automatically.
- Reject negative precipitation unless the caller explicitly requests a documented clipping policy upstream.
- Do not silently interpret missing values as valid zeros. The current inference implementation fills non-finite values with zero, so warn and stop when validation finds them in production inputs.
- Preserve chronological alignment and the exact trained spatial grid. Shape compatibility alone does not prove coordinate alignment.
- Use `--enforce-temperature-order` unless reproducing a historical run that intentionally omitted it.
- Do not submit scheduler jobs or overwrite existing outputs without explicit authorization. Use `$climate-downscaling-ops` for Slurm operations.
- Use `$climate-downscaling-evaluation` to calculate metrics or spatial statistics.

## Output

Return a concise run record containing route, input files, checkpoint(s), command, date/index interval, output location, output shape, and warnings. Never claim scientific validity from successful execution alone.
