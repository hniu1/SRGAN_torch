# Demo preparation validation

Prepared on branch `GM_Downscaling_Demo1` for Frontier.

- All 12 unit/integration tests passed, including a relocated 6× NetCDF inference
  run, Daymet variable aliases, output dates/shapes, and overwrite protection.
- All 365 days of each bundled coarse input passed strict missing-value, units,
  and precipitation-range validation. Coordinates/time agree across variables.
- The pretrained checkpoint hash matches the original model registry. Frozen
  normalization matches the checkpoint's training metadata.
- The real checkpoint produced finite `(1, 3, 72, 72)` output on a small Daymet
  patch on CPU. This is a patch smoke test, not a full-domain benchmark.
- All three skills passed the skill validator. Shell syntax, JSON parsing,
  documentation links and `git diff --check` passed.
- Binary asset SHA-256 values and original source paths are recorded in
  `daymet/asset-manifest.json`.

Full-domain GPU inference and Slurm submission have not been run as part of this
preparation. Historical full-year metrics are copied reference results, not new
measurements. The first team run should submit the one-day inference launcher
and verify the output metadata and fields.

Old ignored artifacts and the legacy archive remain in the source workspace;
they are excluded from the handoff tar. They are not protected by Git branches.
The source changes are uncommitted; no remote push was performed.
