# Bundled Daymet data

The demo contains byte-for-byte copies of six daily 1990 NetCDF files:
`Daymet_ERA5_{tmin,tmax,prcp}_dy_1990_{0p25deg,trim}.nc`.
The `0p25deg` files are model inputs; `trim` files are native 1/24° truth.
Do not use pre-interpolated `0p25degto0p0416deg` files as truth.

Original Frontier sources:

- Data: `/lustre/orion/cli138/proj-shared/dr6/NA-Downscaling/data`
- DEM: `/lustre/orion/cli138/proj-shared/dr6/NA-Downscaling/DEM/final-elev`
- Checkpoint and prepared static arrays: the original `artifacts/runs/refine_stage2_v1/best.pt`
  and `artifacts/data/daymet_mv_stage2_1980_1990/shared` in the source workspace.

`asset-manifest.json` records source paths, packaged paths, sizes and SHA-256
checksums. To restore ignored assets after cloning on this same Frontier system,
run `python scripts/restore_demo.py`, then `python scripts/check_demo.py`.
Restoration requires those original paths to remain accessible. The complete
handoff archive does not need the originals.

Only the 365-day test split is bundled. Training and validation are intentionally
absent. Normalization values were fitted on 1980–1987 and are preserved exactly
as used by the pretrained model. Never refit them on the demo year.
