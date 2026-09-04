# Climate downscaling agent authoring kit

This folder is the framework-neutral handoff package for teams building their own agent skills or tools around the ClimateSwin pipeline. It describes stable capabilities and contracts without requiring Codex's skill format.

Ready-to-use Codex skill packages live in [`../skills`](../skills). The two folders have different purposes:

- `skill-authoring-kit/`: source specifications, schemas, examples, and templates to adapt to any agent framework.
- `skills/`: installable instructions and helper scripts that an agent can use directly.

## Package map

| File | Purpose |
|---|---|
| `capability-catalog.md` | Skill boundaries, triggers, inputs, and outputs |
| `data-contract.md` | Variables, units, grids, quality rules, and scale transitions |
| `interface-contract.md` | Recommended request/result/error protocol |
| `operations-and-permissions.md` | Read-only versus approval-required actions |
| `model-registry.json` | Machine-readable stage/checkpoint/metric registry |
| `schemas/*.schema.json` | JSON validation schemas for agent requests and results |
| `examples/*.json` | Concrete protocol examples |
| `templates/*` | Starting points for another agent framework |

## Recommended integration

1. Expose three separate capabilities: downscaling inference, evaluation, and HPC operations.
2. Validate requests against `schemas/downscaling-request.schema.json` before generating commands.
3. Validate input data against `data-contract.md` and the selected registry stage.
4. Make filesystem inspection and Slurm status read-only tools.
5. Put job submission, cancellation, overwrite, and deletion behind explicit human approval.
6. Emit `schemas/downscaling-result.schema.json` records for traceability.
7. Pin checkpoint hashes in the deployment configuration and update the registry only after evaluation.

Paths in the registry are relative to the repository root unless marked otherwise. Site-specific environment paths belong in deployment configuration, not in the scientific request.
