# Data contract

The maintained scientific and path contract is in
[the inference skill](../skills/refine-downscaling/references/data-contract.md).
Asset provenance and restoration are in [daymet/README.md](../daymet/README.md).
Shared Frontier data are already available at
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/daymet`.
Models are in the same demo root's `artifacts/` directory. Use
`daymet/prepared/manifest.json` with `artifacts/runs/refine_stage2_v1/best.pt`
when running there. The registry and portable examples retain the equivalent
`checkpoints/refine_6x.pt` copy used by the packaged default layout.
