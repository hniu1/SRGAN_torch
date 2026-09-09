# Checkpoint selection

Use `checkpoints/refine_6x.pt` with `daymet/prepared/manifest.json`.
The authoritative hash, dimensions and historical metrics are in
`skill-authoring-kit/model-registry.json`. Verify with `scripts/check_demo.py`.
Load only trusted checkpoints: the existing training format uses Python pickle.

Preserve the released checkpoint and normalization. A replacement must have
6× reconstruction, matching variable order, geometry and transform parameters.
The internal historical stage label is 2; no preceding model is needed.
