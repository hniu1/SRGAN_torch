# Agent integration kit

This package describes the sole REFINE 6× inference route on Frontier.
Start with the repository README, then use:

- `model-registry.json`: checkpoint identity, grid and reference metrics.
- `data-contract.md`: input/output requirements.
- `interface-contract.md`: structured request and result fields.
- `schemas/`: JSON schemas (the legacy identifier `stage2` means the single 6× model).
- `examples/`: illustrative requests/results, not execution records.
- `capability-catalog.md`: routing to maintained entry points and skills.
- `operations-and-permissions.md`: scheduler and artifact authorization.
- `templates/`: optional starting point for team-specific skills.

The ready-to-use skills are in `../skills/`. Training guidance remains in
`../skills/refine-ops/references/training.md`.
