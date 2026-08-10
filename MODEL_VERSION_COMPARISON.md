# Stage-1 SRGAN Patch Downscaling: Model Version Comparison

**Task:** daily `tmax` downscaling from 1° to 0.25° over North America  
**Scale factor:** 4×  
**Training period:** 1980–1989 Daymet/ERA5 (3,653 days)  
**Independent evaluation period:** 1990 Daymet/ERA5 (365 days)

This document compares the completed first patch model with the new
terrain-aware patch model. It is written so that the tables and short sections
can be copied directly into presentation slides.

---

## 1. Motivation

The earlier deep generator trained on 8×8 patches produced a strong rectangular
frame during full-domain inference. Its receptive field was much larger than
the training patch and its many zero-padded convolutions made patch boundaries
part of the learned signal.

The first revised patch model removed that frame by using:

- reflection padding instead of zero padding;
- a shallow generator matched to small patches;
- no BatchNorm;
- a bilinear temperature baseline plus a learned correction;
- PixelShuffle upsampling;
- generator pretraining before adversarial training; and
- ten years of training data rather than one year.

The remaining 1990 errors concentrate over western terrain, complex coastlines,
Alaska, and the apparent Daymet/ERA5 transition near Canada. This motivated a
second model with native 0.25° elevation and more capacity.

---

## 2. Versions at a Glance

| Property | Version 1: patch model | Version 2: terrain-aware model |
|---|---|---|
| Model version | `tmax_stage1_patch_pixelshuffle_10yr` | `tmax_stage1_patch_hr_elev_deep_10yr` |
| Status | Trained and evaluated | Trained and evaluated |
| Training patch | 8×8 LR → 32×32 HR | 16×16 LR → 64×64 HR |
| Inference sizes | Flexible | Flexible, including 8×8 and full domain |
| LR inputs | 1° tmax, 1° elevation | 1° tmax, 1° elevation |
| HR inputs | None | 0.25° elevation, upscaled LR elevation, elevation anomaly |
| Feature channels | 64 | 96 |
| LR residual blocks | 1 | 4 |
| HR fusion residual blocks | 0 | 2 |
| Upsampling | PixelShuffle ×2, twice | PixelShuffle ×2, twice |
| Padding | Reflection | Reflection |
| BatchNorm | No | No |
| Output formulation | Bilinear baseline + CNN correction | Bilinear baseline + terrain-aware CNN correction |
| Generator parameters | ~408,000 | ~1,874,000 |
| Training patches/day/epoch | 16 | 16 |
| Prepared training days | 2,922 | 2,922 |
| Held-out days in training cache | 731 | 731 |
| External test year | 1990 | 1990 after training |

The previous ~20-million-parameter generator is not used in either revised
version.

---

## 3. Version 1 Architecture

```text
1° tmax + 1° elevation
          ↓
reflection-padded stem convolution
          ↓
one residual block
          ↓
PixelShuffle ×2
          ↓
PixelShuffle ×2
          ↓
learned temperature correction
          +
bilinearly upscaled 1° tmax
          ↓
0.25° tmax prediction
```

The model is fully convolutional. Training used 8×8 patches, but inference can
accept 8×8, 16×16, or the complete 57×129 LR domain without dense-layer size
constraints or patch stitching.

---

## 4. Version 1 Training Outcome

Generator pretraining stopped after 23 epochs. Adversarial training stopped
after 21 epochs. Validation favored the pretrained generator checkpoint:

| Checkpoint | Held-out patch MAE | Held-out patch RMSE | Bias |
|---|---:|---:|---:|
| `g_init.pth` | 0.183 °C | 0.416 °C | +0.008 °C |
| `g.pth` | 0.190 °C | 0.418 °C | −0.001 °C |

The GAN stage did not improve numerical temperature accuracy. Consequently,
the independent 1990 evaluation used `g_init.pth`.

### Boundary result

The rectangular frame was removed. On held-out patches:

| Region | MAE |
|---|---:|
| Outer four HR pixels | ~0.22 °C |
| Patch interior | ~0.16 °C |

Some edge degradation remains, but there is no systematic square-frame pattern.

![Held-out patch frame check](output/tmax_stage1_patch_pixelshuffle_10yr/plots/heldout_patch_frame_check.png)

---

## 5. Independent 1990 Daymet Evaluation

The correct paired evaluation is:

```text
Daymet/ERA5 1990 at 1°
          ↓ model
predicted 1990 tmax at 0.25°
          ↓ compare same date
Daymet/ERA5 1990 at 0.25°
```

### Full-year daily metrics

| Method | Bias | MAE | RMSE |
|---|---:|---:|---:|
| Bilinear interpolation | −0.013 °C | 0.500 °C | 1.437 °C |
| Version 1 SRGAN | **−0.005 °C** | **0.473 °C** | **1.377 °C** |

Version 1 improved daily MAE over bilinear interpolation by **5.3%**.

### Temporal-mean field

| Metric | Version 1 SRGAN |
|---|---:|
| Mean-field bias | −0.005 °C |
| Mean-field MAE | 0.323 °C |
| Mean-field RMSE | 0.890 °C |

![1990 temporal means](output/tmax_stage1_patch_pixelshuffle_10yr/daymet_1990_evaluation/plots/temporal_mean_prediction_truth_difference.png)

![1990 bias, MAE, and RMSE](output/tmax_stage1_patch_pixelshuffle_10yr/daymet_1990_evaluation/plots/bias_mae_rmse_maps.png)

![1990 daily metrics](output/tmax_stage1_patch_pixelshuffle_10yr/daymet_1990_evaluation/plots/daily_metrics.png)

### Interpretation

Version 1 is stable, nearly unbiased, and better than bilinear interpolation,
but the added value is modest. The largest persistent errors follow western
mountain ranges, coastlines, Alaska, and northern/source-boundary features.

---

## 6. Terrain Diagnosis

Version 1 receives only the 1° DEM. It cannot see the exact mountain, ridge,
valley, or coastal elevation inside each 1° grid cell.

The difference between native 0.25° elevation and bilinearly upscaled 1°
elevation has the following magnitude:

| Statistic | Unresolved elevation |
|---|---:|
| Mean absolute difference | 34 m |
| 95th percentile | 186 m |
| Maximum | 1,969 m |

Diagnostic correlations over the annual-mean field were:

| Relationship | Correlation |
|---|---:|
| True temperature correction vs. unresolved elevation | −0.467 |
| Absolute Version 1 error vs. absolute unresolved elevation | +0.456 |
| Version 1 error vs. unresolved elevation | +0.391 |

These values provide direct evidence that unresolved topography explains a
substantial part of the remaining spatial error.

---

## 7. Why Full-Domain Training Could Previously Memorize Geography

In older full-domain training, every grid index always represented the same
location. A deep network with a large receptive field and boundary cues could
partially memorize static geographic corrections even without native HR
elevation.

Random patch training changes the problem:

```text
same local 1° temperature and elevation
may correspond to different unresolved 0.25° terrain
at different geographic locations
```

The patch pairs are correctly aligned—the HR origin is exactly four times the
LR origin. The ambiguity comes from missing fine-resolution and absolute
location information, not an LR/HR crop-alignment bug.

---

## 8. Version 2 Terrain-Aware Design

Version 2 retains the stable residual formulation while adding terrain at the
resolution where temperature is predicted.

```text
LR branch:
1° tmax + 1° elevation
    → 4 residual blocks
    → PixelShuffle ×4

HR terrain branch:
0.25° elevation
upscaled 1° elevation
elevation anomaly = HR elevation − upscaled LR elevation

Fusion:
HR image features + terrain predictors
    → 2 HR residual blocks
    → temperature correction
    + bilinear tmax baseline
    → 0.25° prediction
```

The elevation anomaly explicitly identifies sub-grid peaks and valleys relative
to the elevation represented by the LR temperature. The final correction layer
starts at zero, so initial predictions equal bilinear interpolation rather than
random temperature fields.

### Version isolation

Version 2 has separate training and evaluation entry points:

- `pipeline_02_train_stage1_hr_elev.py`
- `pipeline_02_train_stage1_hr_elev.slurm`
- `pipeline_03_evaluate_daymet_1990_hr_elev.slurm`

The original Version 1 trainer and checkpoints remain unchanged.

The Version 2 output directory reuses the 1.8 GB prepared cache through symbolic
links and adds a static 228×516 HR elevation field. No ten-year preparation was
rerun and no large arrays were duplicated.

---

## 9. Sampling Coverage and Current Limitation

For Version 2, a 16×16 patch can start at:

```text
42 row positions × 114 column positions = 4,788 origins/day
```

Across 2,922 training days:

```text
13,990,536 possible day-location pairs
46,752 random draws/epoch
0.334% of all day-location pairs/epoch
```

Because draws are distributed across 2,922 days, nearly every geographic origin
is represented by some day in a typical epoch. However, not every date-location
combination is covered.

Expected unique origin coverage for each individual day is:

| Epochs | Expected unique origins/day | Coverage |
|---:|---:|---:|
| 1 | ~16 | 0.33% |
| 23 | ~354 | 7.4% |
| 50 | ~737 | 15.4% |
| 100 | ~1,360 | 28.4% |

The current Version 2 validation configuration uses only four deterministic
patches per held-out day. Those positions are not geographically representative.
Before relying on validation-based checkpoint selection, validation should be
expanded to an approximately 8×8 spatial grid (64 patches/day), ideally with
additional mountain, high-gradient, coastline, and source-boundary sampling.

---

## 10. Remaining Canadian-Border Question

The sharp northern feature may be partly caused by the merged Daymet/ERA5 target
rather than model capacity. Daymet and ERA5 differ in native resolution, terrain,
land masks, interpolation, and bias characteristics.

Native HR elevation should improve physically driven terrain errors, but it
cannot infer an artificial data-source transition. If the line remains in
Version 2, the next additions should be:

1. a Daymet-versus-ERA5 source mask;
2. latitude and longitude channels;
3. a valid land/data mask; or
4. a smoothly blended or spatially consistent target product.

---

## 11. Terrain-Aware Training Comparison

Two versions use the same 1.87-million-parameter terrain-aware generator and
the same independent 1990 evaluation data. The stable version changes the
training procedure rather than the generator architecture.

| Property | Earlier deep run | Stable run |
|---|---|---|
| Model version | `tmax_stage1_patch_hr_elev_deep_10yr` | `tmax_stage1_patch_hr_elev_stable_10yr` |
| Evaluated checkpoint | `g_init.pth` | `g.pth` (best validation epoch 47) |
| Random training patches/day | 16 | 64 |
| Validation patches/day | 4 | 64 |
| Generator objective | Strongly content-dominated | MSE + 0.1 MAE + 0.1 gradient + 1e-4 GAN |
| Gradient clipping | No | Norm 1.0 |
| Generator learning rate | Higher legacy setting | 2e-5 |
| Discriminator learning rate | Higher legacy setting | 1e-5 |

The checkpoint difference is important: this table compares the earlier
pretrained generator with the best adversarial checkpoint from the stable run.
Additional cross-evaluation of both `g_init.pth` and `g.pth` would isolate the
effect of GAN fine-tuning from the other training changes.

### Independent 1990 metrics

| Metric | Bilinear | Earlier deep | Stable | Stable vs. earlier |
|---|---:|---:|---:|---:|
| Daily bias | −0.013 °C | +0.0077 °C | +0.0065 °C | Similar, near zero |
| Daily MAE | 0.4996 °C | 0.4698 °C | **0.2117 °C** | **54.9% lower** |
| Daily RMSE | 1.4367 °C | 1.3618 °C | **0.4502 °C** | **66.9% lower** |
| Temporal-mean bias | — | +0.0077 °C | +0.0065 °C | Similar, near zero |
| Temporal-mean MAE | — | 0.3047 °C | **0.0668 °C** | **78.1% lower** |
| Temporal-mean RMSE | — | 0.8478 °C | **0.1625 °C** | **80.8% lower** |
| Daily MAE improvement over bilinear | — | 6.0% | **57.6%** | +51.7 percentage points |

### Temporal-mean prediction, truth, and difference

| Earlier terrain-aware deep run | Stable terrain-aware run |
|---|---|
| ![Earlier deep temporal-mean comparison](output/tmax_stage1_patch_hr_elev_deep_10yr/daymet_1990_evaluation/plots/temporal_mean_prediction_truth_difference.png) | ![Stable temporal-mean comparison](output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/plots/temporal_mean_prediction_truth_difference.png) |

The stable model removes most of the broad persistent error. Its remaining
temporal-mean errors are concentrated over steep western terrain, coastlines,
and northern or source-data boundaries.

For the stable model, the absolute temporal-mean error distribution is:

| Statistic | Absolute error |
|---|---:|
| Median | 0.0176 °C |
| 90th percentile | 0.1711 °C |
| 95th percentile | 0.2706 °C |
| 99th percentile | 0.6478 °C |
| Cells within ±0.4 °C | 97.4% |
| Cells exceeding ±1.0 °C | 0.39% |

The difference color scale uses the 98th percentile, so a small number of
localized errors exceed the displayed ±0.4 °C range.

### Spatial bias, MAE, and RMSE

| Earlier terrain-aware deep run | Stable terrain-aware run |
|---|---|
| ![Earlier deep spatial error maps](output/tmax_stage1_patch_hr_elev_deep_10yr/daymet_1990_evaluation/plots/bias_mae_rmse_maps.png) | ![Stable spatial error maps](output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/plots/bias_mae_rmse_maps.png) |

### Daily metrics

| Earlier terrain-aware deep run | Stable terrain-aware run |
|---|---|
| ![Earlier deep daily metrics](output/tmax_stage1_patch_hr_elev_deep_10yr/daymet_1990_evaluation/plots/daily_metrics.png) | ![Stable daily metrics](output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/plots/daily_metrics.png) |

### Result files

- Earlier summary: `output/tmax_stage1_patch_hr_elev_deep_10yr/daymet_1990_evaluation/evaluation_summary.json`
- Stable summary: `output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/evaluation_summary.json`
- Both directories also contain the full 365-day prediction array, temporal-mean
  prediction and truth arrays, spatial error maps, and daily metric plots.

---

## 12. Conclusions

### Version 1 achievements

- Removed the artificial rectangular frame.
- Reduced catastrophic bias and stabilized full-domain inference.
- Achieved 0.473 °C daily MAE on unseen 1990 Daymet/ERA5.
- Improved MAE over bilinear interpolation by 5.3%.
- Showed that content pretraining outperformed GAN fine-tuning for numerical
  temperature accuracy.

### Version 1 limitation

- Only 1° elevation was available to predict 0.25° terrain-driven temperature.
- Remaining errors strongly correlate with unresolved HR terrain.
- Random patches remove the accidental geographic memorization available to a
  fixed-domain deep model.

### Terrain-aware stable-model outcome

- Native HR elevation, denser patch sampling, gradient-aware loss, and stable
  optimization reduced independent-1990 daily MAE to 0.212 °C.
- The temporal-mean MAE fell to 0.067 °C, with 97.4% of cells within ±0.4 °C.
- No artificial rectangular patch frame is visible in full-domain inference.
- Remaining localized errors still follow steep terrain, complex coastlines,
  and northern/source-data boundaries; masks or coordinate channels remain
  reasonable future experiments.

### Scientific evaluation principle

- Use paired Daymet 1° and Daymet 0.25° dates for daily accuracy.
- Use GCM-versus-Daymet comparisons only for climatologies and distributions;
  a free-running GCM day is not synchronized with observed weather.
