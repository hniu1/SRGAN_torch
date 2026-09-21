# Half-year technical progress: North American climate downscaling

**Leadership and sponsor briefing • Prepared September 21, 2026**  
**Reporting window:** March 21–September 21, 2026. Available branch history documents the milestones below from June through September; it does not establish a complete month-by-month record for the entire window.  
**Branches reviewed:** `SRGAN_patch_training` (`e49aa97`, August 10) and `variable_fusion_patch` (`29003b3`, September 8). The latter is the locally available branch corresponding to the requested “variable_fusion_training” work.

## Executive summary — ready to paste into a slide

- **Made continental-domain training practical:** aligned random patches and on-demand data loading reduce the spatial footprint of each training sample and avoid loading the complete multi-year archive into memory.
- **Changed the prediction task to learning corrections:** models start from the upscaled low-resolution climate field and learn the fine-resolution adjustment, retaining a direct connection to the input climate signal.
- **Improved terrain representation and training stability:** high-resolution elevation, denser sampling, and revised optimization reduced independent-test maximum-temperature MAE from **0.473°C to 0.212°C**.
- **Advanced to joint multivariable transformer downscaling:** REFINE combines minimum temperature, maximum temperature, and precipitation through variable attention and a shared SwinV2 spatial encoder. Stage-1 maximum-temperature MAE reaches **0.201°C**, with approximately **60% lower temperature MAE** and **39% lower precipitation MAE** than bilinear interpolation.
- **Extended the framework to finer resolution:** a separately evaluated 0.25° → 1/24° stage reduces temperature MAE by approximately **75–76%** and precipitation MAE by **63%** relative to its own bilinear baseline.

**Suggested sponsor message:** “We progressed from memory-constrained continental training to a terrain-aware, multivariable downscaling framework. The system learns local refinements to coarse climate fields and has demonstrated substantial gains over interpolation on an independent test year. The next milestone is validating the complete two-stage pipeline and its performance on extremes and climate-model inputs.”

## Slide 1 — Patch training addresses the North American memory bottleneck

**Main message:** Train on small, aligned pieces of the continent while retaining the ability to predict full spatial fields.

- The Stage-1 domain is **57 × 129 low-resolution cells**, mapping to **228 × 516 high-resolution cells**. Full-domain training retains intermediate feature maps and gradients across that entire area.
- The initial patch model uses **8 × 8 → 32 × 32** samples; the terrain-aware CNN uses **16 × 16 → 64 × 64** samples. The fine-resolution crop is aligned exactly with its coarse-resolution input.
- Memory-mapped arrays and lazy sampling read and normalize only the requested crops. This addresses host-memory pressure as well as the GPU activation footprint.
- REFINE adds a surrounding context region: **24 × 24 → 96 × 96** for Stage 1, with loss computed only over the central **16 × 16 → 64 × 64** core. Real neighboring context reduces the influence of artificial patch edges.
- Stage 2 reads aligned patches directly from NetCDF using a lightweight index, avoiding duplication of the roughly 200 GB uncompressed daily archive described in the branch documentation.

**Quantitative illustration:** An 8 × 8 patch has about **115 times fewer spatial cells** than the 57 × 129 full domain; a 16 × 16 patch has about **29 times fewer**. These are geometric ratios per sample, **not measured GPU-memory or speedup factors**. Architecture, batch size, attention, and optimizer state also determine memory use.

**Leadership implication:** Continental coverage and larger training archives become compatible with bounded training samples. Both REFINE run configurations record eight-GPU training with mixed precision; no controlled before/after peak-memory benchmark was found.

## Slide 2 — Learn the update to the input field

**Main message:** Preserve the coarse climate structure and learn the additional spatial detail.

```text
Low-resolution climate field ── bilinear upsampling ─────────────┐
                                                              + → Fine-resolution prediction
Climate variables + terrain + context ── learned correction ───┘
```

- The prediction is **upscaled input + learned correction**. The output correction layers are initialized to zero, so training begins from the interpolation baseline.
- The network can focus on unresolved terrain effects and local spatial structure while retaining a direct input-to-output path.
- This formulation is present in the revised patch CNNs and retained in REFINE. It does not, by itself, guarantee conservation or physical consistency.
- “Learning updates instead of generating from scratch” refers to this output formulation. The earlier SRGAN was also conditioned on low-resolution inputs; it was not unconditional generation from random noise.

**Technical note for speaker questions:** REFINE performs this addition in transformed model space. Temperature uses standard scaling; precipitation uses standardized `log1p`. Therefore, a zero precipitation correction corresponds to interpolation in transformed space, whereas the reported bilinear benchmark interpolates physical precipitation directly.

## Slide 3 — Terrain and training design delivered the largest temperature gain

**Main message:** Fine-resolution geographic information and stable training were central to the accuracy improvement.

- Reflection padding, a shallower initial patch generator, removal of BatchNorm, and PixelShuffle addressed the rectangular boundary artifact documented in earlier patch experiments.
- Native high-resolution elevation and the difference between fine and upscaled coarse elevation expose ridges and valleys that the low-resolution input cannot resolve.
- The stable terrain-aware run increased random training patches per day from **16 to 64** and validation patches from **4 to 64**. It also used gradient-aware loss, lower learning rates, and gradient clipping.
- On independent 1990 data, daily maximum-temperature MAE fell from **0.4698°C to 0.2117°C** between the earlier and stable terrain-aware runs: **54.9% lower**. Annual-mean-field MAE fell from **0.3047°C to 0.0668°C**.

**Interpretation:** Adding terrain and depth alone produced only a small improvement in the earlier run. The larger gain followed the combined sampling and optimization changes. These runs do not isolate the contribution of each change; the earlier terrain-aware result used a pretrained generator and the stable result used an adversarially fine-tuned checkpoint.

## Slide 4 — Multivariable fusion and transformer advancement

**Main message:** REFINE learns shared climate information while retaining specialized outputs for each variable.

- **Variable fusion:** separate input encoders and learned variable identities feed cross-variable attention at each coarse-grid location, combining `tmin`, `tmax`, and precipitation into a shared representation.
- **Spatial transformer:** a SwinV2 encoder alternates regular and shifted **8 × 8 attention windows**. Shifting windows allows information exchange across window boundaries without full-domain global attention.
- **Geographic and seasonal context:** elevation, coordinates, valid-data masks, and seasonal features condition the shared representation; native fine-resolution terrain is fused during reconstruction.
- **Specialized outputs:** a separate decoder learns the correction for each output variable. Stage 1 uses **2× followed by 2×** PixelShuffle; Stage 2 uses **2× followed by 3×**.
- **Training objectives:** per-variable data and gradient losses are supplemented by temperature-order and coarse-grid precipitation-conservation penalties. These penalties encourage consistency; they are not exact guarantees.

The recorded runs use **six groups of six transformer blocks**, 96 features, and six spatial attention heads. Stage 1 has **4.77 million parameters**; Stage 2 has **5.18 million**. REFINE is deterministic and **does not use an adversarial discriminator**. Variable dropout is implemented as a training feature; improved robustness to missing variables has not been established by the available comparisons.

**Leadership implication:** One shared model now produces three climate variables, expanding capability beyond the single-temperature baseline. Controlled experiments are still needed to distinguish the benefits of variable fusion from the transformer and other concurrent changes.

## Slide 5 — Comparable Stage-1 results across branches

**Task:** daily maximum temperature, **1° → 0.25°**, North America; all **365 days of 1990**. Lower MAE and RMSE are better. MAE measures average absolute error; RMSE places more weight on large errors.

| Method | Bias (°C) | MAE (°C) ↓ | RMSE (°C) ↓ | MAE reduction vs. bilinear |
|---|---:|---:|---:|---:|
| Bilinear interpolation | -0.0130 | 0.4996 | 1.4367 | 0.0% |
| Initial patch CNN | -0.0053 | 0.4731 | 1.3775 | 5.3% |
| Earlier terrain-aware CNN | +0.0077 | 0.4698 | 1.3618 | 6.0% |
| Stable terrain-aware SRGAN | +0.0065 | 0.2117 | 0.4502 | 57.6% |
| REFINE: fusion + SwinV2 | +0.0048 | 0.2007 | 0.4306 | 59.8% |

**Key result:** REFINE reduces maximum-temperature MAE by **5.17%** and RMSE by **4.35%** relative to the strongest saved SRGAN baseline. The additional MAE gain is about **0.011°C**. Its larger capability advance is jointly predicting three variables while achieving this further temperature improvement.

**Comparison qualifications:** Both evaluations use the same nominal 1990 task, matching grid-cell counts (42,941,520) and essentially identical maximum-temperature bilinear scores. However, this is a comparison of completed systems, not a controlled architecture experiment. SRGAN used a random 80/20 split of 1980–1989; REFINE trained on 1980–1987 and validated on 1988–1989. Objectives, sampling, model size, checkpoint selection, and output correction also differ. Statistical significance and variation across training seeds were not evaluated here.

## Slide 6 — Stage-1 multivariable performance

**Task:** 1° → 0.25°; independent 1990 evaluation, 42,941,520 valid space-time cells per variable.

| Variable | Bilinear MAE | REFINE MAE ↓ | Bilinear RMSE | REFINE RMSE ↓ | MAE reduction |
|---|---:|---:|---:|---:|---:|
| tmin (°C) | 0.5215 | 0.2097 | 1.6703 | 0.4507 | 59.8% |
| tmax (°C) | 0.4996 | 0.2007 | 1.4367 | 0.4306 | 59.8% |
| prcp (mm/day) | 0.4531 | 0.2784 | 1.4264 | 0.9912 | 38.6% |

**Slide takeaway:** “Joint downscaling reduces daily temperature MAE by about 60% and precipitation MAE by about 39% relative to interpolation.”

Aggregate biases are −0.0042°C for minimum temperature, +0.0048°C for maximum temperature, and −0.0046 mm/day for precipitation. Small aggregate bias does not establish small errors everywhere or accurate extremes.

**Evaluation detail:** The saved evaluation applies a temperature-order correction. Before that correction, **1.534%** of valid temperature comparisons had predicted minimum temperature above maximum temperature. The reported temperature accuracy therefore describes the model plus this correction.

## Slide 7 — Extension to 1/24° resolution

**Main message:** The same shared representation supports a second, finer reconstruction stage.

- Stage 2 maps **0.25° → 1/24°**, a sixfold increase in grid resolution along each spatial dimension.
- It reuses compatible Stage-1 encoder weights and trains new upsampling and decoder components; the recorded initialization loaded all **646 eligible tensors**.
- Training uses **12 × 12** coarse context patches and supervises the central **8 × 8 → 48 × 48** region. Both saved REFINE histories contain 100 epochs.

**Independent Stage-2 results:** 365 days of 1990; 1,545,894,720 valid space-time cells per variable.

| Variable | Bilinear MAE | REFINE MAE ↓ | Bilinear RMSE | REFINE RMSE ↓ | MAE reduction |
|---|---:|---:|---:|---:|---:|
| tmin (°C) | 0.2016 | 0.0497 | 0.9939 | 0.1680 | 75.4% |
| tmax (°C) | 0.2012 | 0.0487 | 0.8497 | 0.1535 | 75.8% |
| prcp (mm/day) | 0.1347 | 0.0495 | 0.5890 | 0.3137 | 63.2% |

**Essential scope note:** These metrics evaluate Stage 2 using **observed 0.25° inputs**. They do not measure the complete **1° → 0.25° → 1/24°** cascade driven by Stage-1 predictions. The smaller Stage-2 errors must not be presented as a further percentage improvement over Stage-1 errors: the inputs, target grid, and baseline differ. Stage-2 temperature scores also include ordering correction; the pre-correction violation rate is **1.415%**.

## Slide 8 — Next milestones for sponsor reporting

1. **Validate the complete cascade:** evaluate Stage 2 on Stage-1 predictions, and test fine-tuning with a mixture of observed and generated intermediate inputs.
2. **Separate the sources of improvement:** compare single-variable and multivariable models, CNN and transformer encoders, and terrain/context options using matched splits, budgets, and postprocessing.
3. **Broaden scientific validation:** test additional independent years, seasonal and regional errors, wet-day behavior, precipitation extremes, and joint-variable consistency. Prioritize mountains, coastlines, and northern/source-data boundaries identified in existing diagnostics.
4. **Measure operational benefits:** record peak host/GPU memory, throughput, inference cost, and sensitivity to patch size. The current evidence supports a memory-conscious design, not a specific measured memory-saving percentage.
5. **Evaluate climate-model applications appropriately:** compare climatologies, distributions, seasonal cycles, and extremes. A free-running GCM day is not synchronized with observed weather, so same-date GCM-versus-Daymet daily differences are not forecast accuracy.

## Suggested figures for the presentation

- **Model progression:** use the Stage-1 comparison table above as the primary quantitative slide.
- **Terrain-aware baseline:** [stable SRGAN annual-mean comparison](../../archive/srgan_v2/output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/plots/temporal_mean_prediction_truth_difference.png).
- **Transformer temperature output:** [Stage-1 maximum-temperature annual means and bias](../../artifacts/runs/refine_stage1_v1/test_1990/spatial_statistics/mean_tmax_comparison.png).
- **Multivariable capability:** [Stage-1 precipitation annual means and bias](../../artifacts/runs/refine_stage1_v1/test_1990/spatial_statistics/mean_prcp_comparison.png).
- **Fine-resolution extension:** [Stage-2 maximum-temperature annual means and bias](../../artifacts/runs/refine_stage2_v1/test_1990/spatial_statistics/mean_tmax_comparison.png).
- **Extreme diagnostics:** [Stage-2 precipitation 95th-percentile map](../../artifacts/runs/refine_stage2_v1/test_1990/spatial_statistics/p95_prcp_comparison.png). Present as a diagnostic; the aggregate MAE tables do not establish extreme-event skill.

When placing maps side by side, verify identical geographic extent and color limits. Annual-mean maps illustrate persistent spatial structure; they do not replace daily error metrics.

## Evidence and provenance

All numerical performance tables were generated from the existing local evaluation JSON files; no training or inference was rerun for this briefing. Relative links require the repository artifacts to remain in place.

| Evidence | Source |
|---|---|
| Initial patch CNN evaluation | [1990 JSON](../../archive/srgan_v2/output/tmax_stage1_patch_pixelshuffle_10yr/daymet_1990_evaluation/evaluation_summary.json) |
| Earlier terrain-aware evaluation | [1990 JSON](../../archive/srgan_v2/output/tmax_stage1_patch_hr_elev_deep_10yr/daymet_1990_evaluation/evaluation_summary.json) |
| Stable terrain-aware evaluation | [1990 JSON](../../archive/srgan_v2/output/tmax_stage1_patch_hr_elev_stable_10yr/daymet_1990_evaluation/evaluation_summary.json) |
| REFINE Stage 1 | [1990 JSON](../../artifacts/runs/refine_stage1_v1/test_1990/evaluation_summary.json), [run configuration](../../artifacts/runs/refine_stage1_v1/run_config.json) |
| REFINE Stage 2 | [1990 JSON](../../artifacts/runs/refine_stage2_v1/test_1990/evaluation_summary.json), [run configuration](../../artifacts/runs/refine_stage2_v1/run_config.json) |
| SRGAN design and experiment context | [model comparison](../../MODEL_VERSION_COMPARISON.md), [patch dataset](../../patch_dataset.py), [generators](../../srgan_torch.py) |
| REFINE architecture and evaluation | `variable_fusion_patch:refine_downscaling/model.py`, `refine_downscaling/transforms.py`, `pipeline_03_evaluate_stage1.py`, `pipeline_07_evaluate_stage2.py`, `README.md` |
| Stage-2 experiment description | `variable_fusion_patch:docs/STAGE2_EVALUATION_1990.md` |

To inspect a source on the other branch without changing the checkout, use, for example, `git show variable_fusion_patch:refine_downscaling/model.py`.

**Milestones in the reviewed history:** June 19 — code import; July 7 — Stage-2 input-size update; August 5 — first documented patch-training version; August 10 — updated terrain-aware pipeline/results; August 27 — Swin transformer implementation; September 1–3 — Stage-2 updates; September 8 — REFINE naming update. Commit dates identify repository milestones, not necessarily the date an experiment completed.
