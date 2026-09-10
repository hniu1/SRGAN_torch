# Inference

Follow the runnable quickstart in the repository `README.md`. On Frontier,
change to `/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1` and
export `MV_CHECKPOINT="$PWD/artifacts/runs/refine_stage2_v1/best.pt"` first.
The Daymet inputs are already in that directory's `daymet/` folder and models
are in `artifacts/`. For direct Python inference, use
`--checkpoint artifacts/runs/refine_stage2_v1/best.pt`.
`sbatch slurm/04_infer.slurm` runs the first day of 1990 on one Frontier GPU.
`MV_END_INDEX=7 sbatch slurm/04_infer.slurm` selects the first week; allocate
sufficient time before increasing to 365 days.

Portable package/default CLI assets are `daymet/prepared`, `checkpoints/refine_6x.pt`, and the three
`daymet/data/*_dy_1990_0p25deg.nc` files. The launcher creates a job-specific
output name under `artifacts/demo/`. The CLI refuses existing output files.
Use `--enforce-temperature-order`, batch size 1, and `--amp` on the GPU.
For other dates/years use the Python CLI with an explicit `--start-date`.

Each output day contains three 1368×3096 float32 fields (about 51 MB uncompressed).
Record the input paths, interval, checkpoint hash, command, output and warnings.
