# Spatial plot standard

Use a horizontal color bar below every panel. Plot longitude and latitude tick labels without `grid x` or `grid y` axis titles. Add Natural Earth country borders; add subtle state/province borders for the United States, Canada, and Mexico only when they remain legible.

## Physical fields

Use `Spectral_r` for `tmin`/`tmax` and `Spectral` for `prcp`.

| Variable | Mean | 95th percentile | 5th percentile |
|---|---:|---:|---:|
| prcp | 0 to 5 | 0 to 25 | 0 to 1 |
| tmax | -5 to 30 | 0 to 40 | -20 to 20 |
| tmin | -20 to 20 | 0 to 50 | -30 to 15 |

## Bias fields

Bias is prediction minus Daymet. Use `RdBu_r` for `tmin`/`tmax` and `RdBu` for `prcp`, centered at zero.

| Variable | Mean | 95th percentile | 5th percentile |
|---|---:|---:|---:|
| prcp | -2 to 2 | -8 to 8 | -0.1 to 0.1 |
| tmax | -2 to 2 | -5 to 5 | -1 to 1 |
| tmin | -2 to 2 | -5 to 5 | -5 to 5 |

Comparison figures should show Daymet, model, and bias with the same physical scale for the first two panels and the stated symmetric bias scale for the third. Include variable, statistic, units, year/split, and stage in the title or caption.
