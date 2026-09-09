# Inference

Follow the runnable quickstart in the repository `README.md`.
`sbatch slurm/04_infer.slurm` runs the first day of 1990 on one Frontier GPU.
`MV_END_INDEX=7 sbatch slurm/04_infer.slurm` selects the first week; allocate
sufficient time before increasing to 365 days.

Default assets are `daymet/prepared`, `checkpoints/refine_6x.pt`, and the three
`daymet/data/*_dy_1990_0p25deg.nc` files. The launcher creates a job-specific
output name under `artifacts/demo/`. The CLI refuses existing output files.
Use `--enforce-temperature-order`, batch size 1, and `--amp` on the GPU.
For other dates/years use the Python CLI with an explicit `--start-date`.

Each output day contains three 1368×3096 float32 fields (about 51 MB uncompressed).
Record the input paths, interval, checkpoint hash, command, output and warnings.
