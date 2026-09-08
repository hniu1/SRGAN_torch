# Metric interpretation

Evaluation aggregates over every valid pixel and evaluated day for each variable.

- **Bias** is mean prediction minus mean target. Positive means the prediction is higher on average.
- **MAE** is mean absolute error and is the primary baseline-comparison statistic.
- **RMSE** weights large errors more strongly than MAE.
- **MAE improvement (%)** is `100 * (baseline_mae - model_mae) / baseline_mae`.
- **Count** is the number of included pixel-time samples and must accompany aggregate metrics.

Temperature errors are reported in degrees Celsius. Precipitation errors are in mm/day.

The `temperature_order_violation_fraction` is computed before optional output enforcement. When enforcement is enabled, identify it explicitly: physically ordered outputs do not mean the unconstrained model never violated `tmin <= tmax`.

## Reference held-out 1990 results

| Stage | Variable | Bias | MAE | RMSE | MAE improvement vs bilinear |
|---|---:|---:|---:|---:|---:|
| 1 | tmin | -0.00424 | 0.20975 | 0.45071 | 59.78% |
| 1 | tmax | 0.00479 | 0.20073 | 0.43065 | 59.82% |
| 1 | prcp | -0.00457 | 0.27845 | 0.99116 | 38.55% |
| 2 | tmin | -0.00102 | 0.04966 | 0.16798 | 75.37% |
| 2 | tmax | 0.00139 | 0.04873 | 0.15347 | 75.78% |
| 2 | prcp | -0.00094 | 0.04954 | 0.31374 | 63.22% |

These are interpolation-style results on the repository's held-out 1990 Daymet split. They are not an end-to-end 1-degree-to-1/24-degree cascade evaluation and should not be presented as such.
