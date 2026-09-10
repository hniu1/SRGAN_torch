# Daymet data locations and provenance

The data are already available on Frontier at
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/daymet`.
Use `data/` for the NetCDF files, `dem/` for terrain, and
`prepared/manifest.json` for the full train/validation/test index. No new download or preparation is
needed for inference or evaluation of the demo.

Models are already available under
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/artifacts/`.
Use `runs/refine_stage2_v1/best.pt` for the 6× inference model and
`runs/refine_stage2_v1/last.pt` for training resumption. The portable package
also includes an identical inference copy at `checkpoints/refine_6x.pt`.

Original Daymet data (Thornton et al., 2021; see the repository README reference)
are provided at 1 km resolution and have been coarsened to approximately
4 km (1/24°) and 25 km (1/4°) for this workflow.

The workspace contains paired daily 1980–1990 NetCDF files:
`Daymet_ERA5_{tmin,tmax,prcp}_dy_YEAR_{0p25deg,trim}.nc`.
The `0p25deg` files are model inputs; `trim` files are the coarsened 1/24° reference fields.
Do not use pre-interpolated `0p25degto0p0416deg` files as truth.

The demo uses the following local paths, relative to the repository root:

- Data: `daymet/data/`
- DEM: `daymet/dem/`
- Prepared static arrays and train/validation/test time indexes: `daymet/prepared/shared/`
- Full Stage 2 manifest: `daymet/prepared/manifest.json`
- Frozen normalization: `daymet/normalization.json`
- Shared pretrained checkpoint: `artifacts/runs/refine_stage2_v1/best.pt`
- Portable pretrained checkpoint copy: `checkpoints/refine_6x.pt`

The complete handoff archive includes these assets and can be moved as a unit.
Run `python scripts/check_demo.py` from the repository root after unpacking.
Large binary assets are ignored by Git, so a Git clone alone is not the complete
demo.

[`asset-manifest.json`](asset-manifest.json) retains the original Frontier source
paths for provenance and optional restoration, together with packaged paths,
sizes and SHA-256 checksums. Inference and evaluation use the local assets above.
Obtain the complete demo archive if local assets are missing. The legacy
`scripts/restore_demo.py` utility explicitly copies from recorded Frontier
sources; it is not part of model execution and must not be used as an automatic
fallback.

`daymet/prepared/` is the canonical full Stage 2 index for every workflow:
training 1980–1987 (2,922 days), validation 1988–1989 (731 days), and test 1990
(365 days). Its source paths resolve to `daymet/data/`, `daymet/dem/`, and
`daymet/normalization.json`. The former
`artifacts/data/daymet_mv_stage2_1980_1990/` location is a compatibility symlink
to this directory in this workspace; new commands use `daymet/prepared/`.

The asset manifest inventories the original 1990 inputs and prepared arrays.
`check_demo.py` additionally checks that paired inputs exist for every split.
The package script includes all files under `daymet/`, including additional
years and the full prepared index. Binary assets remain ignored by Git.

Normalization values were fitted on 1980–1987 and are preserved exactly
as used by the pretrained model. Never refit them on the demo year.
