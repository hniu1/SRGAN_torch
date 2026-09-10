# Input contract

The model requires three aligned daily fields in order `tmin`, `tmax`, `prcp`.
Input grid: 228×516 at 1/4°. Output grid: 1368×3096 at 1/24°.
Aliases include `tmin_dy`, `tmax_dy`, `prcp_dy` in the bundled Daymet files.
Temperatures must be Celsius; precipitation must be mm/day and nonnegative.
Inference does not convert Kelvin. Stop on nonfinite inputs rather than silently
changing the missing-data policy. Preparation/training mask missing fields and
clamp finite negative precipitation under their documented training policy.

Daymet data are already staged at
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/daymet`.
Use its `data/` files and `prepared/manifest.json`; the matching pretrained model
is `artifacts/runs/refine_stage2_v1/best.pt` relative to the shared demo root.
Original 1 km Daymet data (Thornton et al., 2021) were coarsened to approximately
4 km and 25 km for this workflow. Check units, identical time coverage,
and native coordinate alignment; shape alone cannot prove grid identity.
`--start-date 1990-01-01` labels index zero for the bundled files. Arbitrary
calendars and irregular daily intervals are not supported by the inference CLI.
NetCDF outputs carry grid indexes, not geographic lat/lon; recover geography
from the native Daymet truth files for plotting.

Manifest paths are relative to `daymet/prepared/manifest.json`; CLI paths are
relative to the repository root. Never refit frozen normalization on test data.
