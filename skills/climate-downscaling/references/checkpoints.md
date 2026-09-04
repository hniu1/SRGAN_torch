# Checkpoints and manifests

Resolve these paths relative to the repository root unless the caller supplies a reviewed alternative.

| Route | Training manifest | Default checkpoint | Scale |
|---|---|---|---|
| Stage 1 | `artifacts/data/daymet_mv_1980_1990/manifest.json` | `artifacts/runs/climateswin_v1/best.pt` | 4x |
| Stage 2 | `artifacts/data/daymet_mv_stage2_1980_1990/manifest.json` | `artifacts/runs/climateswin_stage2_v1/best.pt` | 6x |

Do not mix a checkpoint with a different manifest. The checkpoint depends on the manifest's variable order, normalization statistics, patch geometry, architecture, and scale factor.

Before using a non-default checkpoint, verify:

1. The variable order is `tmin`, `tmax`, `prcp`.
2. Input and output resolutions match the intended stage.
3. Normalization statistics came from the corresponding training data.
4. The checkpoint loads without missing or unexpected model keys.
5. Evaluation metadata identifies the checkpoint and dataset unambiguously.

For automation or external agents, use `skill-authoring-kit/model-registry.json` as the machine-readable registry.
