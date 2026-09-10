# Frontier environment

The shared demo root is
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1`.
Daymet data are already in `daymet/`; models are in `artifacts/`.
For 6× inference/evaluation, export
`MV_CHECKPOINT="$PWD/artifacts/runs/refine_stage2_v1/best.pt"` after changing
to that root. This overrides the portable `checkpoints/refine_6x.pt` default.

From the repository root, `source scripts/frontier_env.sh` activates the existing
ROCm environment. Override its path using `REFINE_ENV` if needed. The launchers
use allocation cli138, ROCm 6.4.1 and the shared torch_rocm environment.

Create `logs/` before `sbatch`: Slurm opens log paths before running the script.
The launchers accept `MV_BASE_DIR`, `MV_DATA_DIR`, `MV_RUN_DIR`, `MV_CHECKPOINT`.
The inference defaults are `daymet/prepared` and `checkpoints/refine_6x.pt`;
training defaults to `daymet/prepared` and writes under
`artifacts/runs/refine_6x`. Set fresh run/output paths for experiments.

Dependencies are recorded in requirements.txt and requirements-lock.txt.
ROCm drivers, modules and Slurm are supplied by Frontier. Plotting uses the
bundled Natural Earth GeoJSON; Cartopy is not required.
