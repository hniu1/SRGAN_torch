# Data contract

## Canonical variables

| Name | Accepted aliases | Required inference units | Validity rule |
|---|---|---|---|
| `tmin` | `tmin` | degrees Celsius | finite; normally `tmin <= tmax` |
| `tmax` | `tmax` | degrees Celsius | finite; normally `tmax >= tmin` |
| `prcp` | `prcp`, `pr`, `precipitation` | mm/day | finite and nonnegative |

The model consumes all three variables jointly. Files remain separate at the interface: each canonical name maps to a path, and multiple names may point to the same multivariable NetCDF. This preserves extensibility without sacrificing joint modeling.

External names such as `tasmin`, `tasmax`, `tmin_dy`, and `tmax_dy` require an explicit rename/adaptation step before the current inference reader can use them.

## Stage geometry

| Stage | Input grid | Output grid | Scale | Default manifest |
|---|---:|---:|---:|---|
| Stage 1 | 57 x 129 | 228 x 516 | 4x | `artifacts/data/daymet_mv_1980_1990/manifest.json` |
| Stage 2 | 228 x 516 | 1368 x 3096 | 6x | `artifacts/data/daymet_mv_stage2_1980_1990/manifest.json` |
| Cascade | 57 x 129 | 1368 x 3096 | 24x | Stage 1 followed by Stage 2 |

Every variable is a three-dimensional daily array `[time, y, x]` or `[time, lat, lon]`. Variable files must have identical time length/order and the exact trained spatial grid, orientation, extent, mask convention, and coordinate order.

## Units and data quality

- Convert Kelvin to Celsius before inference with `C = K - 273.15`.
- Do not treat negative precipitation as physical data. First determine whether it is a fill value, corruption, interpolation artifact, or unit/encoding problem. Apply an explicit documented clipping policy only after diagnosis.
- Decode `_FillValue`/`missing_value` metadata. Do not silently replace missing cells with zero; the current inference implementation eventually zero-fills non-finite normalized values, which can create boundary artifacts.
- Confirm timestamps represent aligned daily values with a compatible calendar.
- Check `tmin <= tmax` before inference and request output order enforcement as a final safeguard.
- Reject unknown units under strict/production validation.

## Output caveat

The current inference writer stores index dimensions `y` and `x`; geographic coordinates are not sufficient for standalone geolocation in every output. Preserve or attach the source grid/projection metadata in downstream systems.

## Extending variables

Adding a variable is a model-version change, not only an interface change. Update tokenizer/decoder configuration, manifest statistics and order, training data, checkpoints, request schema enumeration, validator aliases/units, evaluation metrics, and the model registry together.
