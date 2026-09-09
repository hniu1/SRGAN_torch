# Input contract

The model requires three aligned daily fields in order `tmin`, `tmax`, `prcp`.
Input grid: 228×516 at 1/4°. Output grid: 1368×3096 at 1/24°.
Aliases include `tmin_dy`, `tmax_dy`, `prcp_dy` in the bundled Daymet files.
Temperatures must be Celsius; precipitation must be mm/day and nonnegative.
Inference does not convert Kelvin. Stop on nonfinite inputs rather than silently
changing the missing-data policy. Preparation/training mask missing fields and
clamp finite negative precipitation under their documented training policy.

Use the copied files under `daymet/data/`. Check units, identical time coverage,
and native coordinate alignment; shape alone cannot prove grid identity.
`--start-date 1990-01-01` labels index zero for the bundled files. Arbitrary
calendars and irregular daily intervals are not supported by the inference CLI.
NetCDF outputs carry grid indexes, not geographic lat/lon; recover geography
from the native Daymet truth files for plotting.

Manifest paths are relative to `daymet/prepared/manifest.json`; CLI paths are
relative to the repository root. Never refit frozen normalization on test data.
