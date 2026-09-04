# Stage-2 ClimateSwin evaluation: 1990

Stage 2 downscales the jointly modeled `tmin`, `tmax`, and precipitation fields
from 0.25 degree to 1/24 degree. The independent test period contains all 365
days of 1990. Metrics use 1,545,894,720 valid space-time cells per variable and
compare the trained model with direct bilinear interpolation of the 0.25-degree
input.

## Performance

| Variable | Model bias | Model MAE | Model RMSE | Bilinear MAE | Bilinear RMSE | MAE improvement |
|---|---:|---:|---:|---:|---:|---:|
| `tmin` (degC) | -0.0010 | 0.0497 | 0.1680 | 0.2016 | 0.9939 | 75.37% |
| `tmax` (degC) | +0.0014 | 0.0487 | 0.1535 | 0.2012 | 0.8497 | 75.78% |
| `prcp` (mm/day) | -0.0009 | 0.0495 | 0.3137 | 0.1347 | 0.5890 | 63.22% |

The model has negligible annual aggregate bias for all three variables. The
temperature RMSE is reduced by 83.1% for `tmin` and 81.9% for `tmax`; the
precipitation RMSE is reduced by 46.7%. Before the optional inference-time
ordering correction, 1.415% of valid cells had `tmin > tmax`; reported metrics
were calculated after that correction was applied.

## Training

Training completed all 100 epochs on eight GPUs in 5 hours 25 minutes. The best
checkpoint is epoch 99 in one-based numbering (stored epoch index 98), with a
validation objective of 0.0070782. The first-epoch validation objective was
0.0142310, so it fell by 50.26% over training.

## Spatial diagnostics

The spatial products are generated under
`artifacts/runs/climateswin_stage2_v1/test_1990/spatial_statistics/`. They show
the model field, Daymet truth, and model-minus-Daymet bias for the annual mean
and selected 5th/95th percentiles. Physical fields use the fixed `Spectral` or
`Spectral_r` scales; model-minus-Daymet panels use the fixed `RdBu` or
`RdBu_r` bias scales supplied for each variable and statistic. The underlying
arrays are retained in `spatial_statistics_1990.npz`.

Maps use native longitude and latitude coordinates without axis-title text.
Natural Earth country borders are overlaid in dark lines, with thinner,
semi-transparent state/province borders for the USA, Canada, and Mexico. Each
panel has its own horizontal color bar directly below the map.

Metrics source: `artifacts/runs/climateswin_stage2_v1/test_1990/evaluation_summary.json`.
