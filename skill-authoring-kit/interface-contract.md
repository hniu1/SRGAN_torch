# Agent interface contract

Use structured requests/results even when the user interacts conversationally. Validate against the schemas in `schemas/`.

## Request

Required fields:

- `request_id`: caller-generated trace identifier
- `action`: `validate`, `infer`, `evaluate`, `status`, `submit`, `resume`, or `cancel`
- `stage`: `stage1`, `stage2`, or `cascade`
- `inputs`: canonical variable-to-path mappings for input-consuming actions
- `output.path`: destination for output-producing actions

Optional fields include `start_date`, `start_index`, `end_index`, `checkpoint`, evaluation controls, execution controls, `job_ids`, and `approval`.

Index ranges are half-open: `start_index` is included and `end_index` is excluded. A cascade request represents two separately traceable inference runs.

## Result

Every result contains:

- request ID and status
- action and stage
- timestamps when execution occurred
- resolved inputs, checkpoint(s), manifest(s), and output(s)
- commands actually executed
- warnings and errors
- artifact metadata when available

Use status values `validated`, `submitted`, `running`, `succeeded`, `failed`, `blocked`, or `cancelled`. Never return `succeeded` merely because request validation passed.

## Error classes

| Code | Meaning | Typical response |
|---|---|---|
| `INVALID_REQUEST` | Schema or missing-field failure | Correct the request |
| `DATA_CONTRACT` | Shape, unit, alignment, missing, or range failure | Repair/regrid input explicitly |
| `CHECKPOINT_MISMATCH` | Stage/model/manifest incompatibility | Select matching registry entry |
| `APPROVAL_REQUIRED` | Mutating action lacks approval | Ask immediately before execution |
| `RESOURCE_FAILURE` | GPU, memory, storage, environment, or scheduler failure | Diagnose logs/state |
| `MODEL_FAILURE` | Loading or inference failure | Preserve logs and partial outputs |
| `EVALUATION_INCOMPATIBLE` | Metrics cannot be compared scientifically | Align datasets/definitions |

Commands and free-text messages are not the source of truth. Persist the structured result beside generated artifacts where practical.

The agent must not create its own approval record. `approval` must come from a trusted user/policy layer, identify the approver and timestamp, and match the exact action scope.
