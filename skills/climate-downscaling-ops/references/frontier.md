# Frontier environment

Run scheduler scripts from the repository root. The checked-in Slurm scripts use:

- Allocation: `cli138`
- ROCm: `6.4.1`
- Miniforge: `23.11.0-0`
- Python environment: `/lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm`
- Logs: `logs/`
- Default Stage 1 data: `artifacts/data/daymet_mv_1980_1990`
- Default Stage 1 run: `artifacts/runs/climateswin_v1`
- Default Stage 2 data: `artifacts/data/daymet_mv_stage2_1980_1990`
- Default Stage 2 run: `artifacts/runs/climateswin_stage2_v1`

Modules loaded by GPU jobs:

```bash
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm
```

Do not assume those versions are portable to another HPC system. Adapt the environment and Slurm account/partition as deployment configuration, without changing scientific contracts.

Useful path overrides are `MV_BASE_DIR`, `MV_DATA_DIR`, `MV_RUN_DIR`, `MV_STAGE2_DATA_DIR`, and `MV_STAGE2_RUN_DIR`. Resolve and report their effective values before submission.
