# Checkpoint selection

For the shared Frontier demo, use
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/artifacts/runs/refine_stage2_v1/best.pt`
with
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/daymet/prepared/manifest.json`.
Models are already staged under `artifacts/`; no training is needed before inference.
From that demo root, pass `--checkpoint artifacts/runs/refine_stage2_v1/best.pt`
to Python or export `MV_CHECKPOINT="$PWD/artifacts/runs/refine_stage2_v1/best.pt"`
before submitting a launcher.

The portable package/default CLI copy is `checkpoints/refine_6x.pt`, containing
the same model weights. `scripts/check_demo.py` verifies that portable copy.
The authoritative hash, dimensions and historical metrics are in
`skill-authoring-kit/model-registry.json`. Verify with `scripts/check_demo.py`.
Load only trusted checkpoints: the existing training format uses Python pickle.

Preserve the released checkpoint and normalization. A replacement must have
6× reconstruction, matching variable order, geometry and transform parameters.
The internal historical stage label is 2; no preceding model is needed.
