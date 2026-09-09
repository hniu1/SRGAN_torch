# Capabilities

| Capability | Implementation | Skill |
|---|---|---|
| Validate demo | scripts/check_demo.py | refine-downscaling |
| Infer 1/4° → 1/24° | pipeline_04_infer.py | refine-downscaling |
| Evaluate and plot | pipeline_03_evaluate.py; utility_plot_spatial_statistics.py | refine-evaluation |
| Prepare/train | pipeline_01_prepare.py; pipeline_02_train.py | refine-ops |
| Frontier jobs | slurm/; skills/refine-ops/scripts/job_status.py | refine-ops |

Inference with the released checkpoint is the demo focus. Training requires
additional non-test years and its own authorized compute allocation.
