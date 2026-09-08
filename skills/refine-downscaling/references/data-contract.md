# Data contract

## Required variables

| Canonical name | Accepted source variable names | Units at inference | Meaning |
|---|---|---|---|
| `tmin` | `tmin` | degrees Celsius | Daily minimum near-surface air temperature |
| `tmax` | `tmax` | degrees Celsius | Daily maximum near-surface air temperature |
| `prcp` | `prcp`, `pr`, `precipitation` | mm/day | Daily precipitation total |

Inputs may be separate NetCDF files or one shared multivariable NetCDF file. Pass one `--input VARIABLE=PATH` argument for each canonical variable. This separation makes adding later variables easier without changing the file layout.

Rename external aliases such as `tasmin`, `tasmax`, `tmin_dy`, or `tmax_dy` before using the current inference pipeline; its reader does not resolve those names.

## Array and grid requirements

- Each variable must resolve to a three-dimensional `[time, y, x]` or `[time, lat, lon]` array.
- All variables must have the same number and ordering of daily time steps.
- Stage 1 input shape is `57 x 129`; expected output is `228 x 516`.
- Stage 2 input shape is `228 x 516`; expected output is `1368 x 3096`.
- Inputs must cover the same geographic grid used by the training manifest, including orientation and cell order.
- There is no implicit regridding, reprojection, longitude wrapping, or calendar conversion.

## Quality rules

- Convert temperature from Kelvin to Celsius upstream: `degC = K - 273.15`.
- Treat negative precipitation as invalid. Diagnose fill values and unit/encoding errors before clipping.
- Decode declared NetCDF fill values as missing data.
- Resolve remaining missing cells upstream with an explicit scientific policy. The current inference path uses zero after normalization for non-finite values, which can create artifacts.
- Confirm `tmin <= tmax` in the input. Use the output enforcement flag as a final physical constraint, not as a substitute for correcting inputs.
- Preserve calendar/date metadata separately when using NumPy output.

The validation script checks names, units, dimensionality, shapes, time counts, non-finite values, negative precipitation, and first-timestep temperature ordering. Use `--strict --full-scan` for production data; full scanning is chunked to limit memory. It does not prove geographic coordinate equality when coordinate variables are absent.
