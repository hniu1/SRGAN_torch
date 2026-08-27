"""Serializable, differentiable transforms for heterogeneous climate variables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


LOG1P_MAX = 15.0


@dataclass(frozen=True)
class TransformSpec:
    name: str
    kind: str
    mean: float
    std: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TransformSpec":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            mean=float(value["mean"]),
            std=float(value["std"]),
        )


def default_transform_kind(variable: str) -> str:
    return "log1p_standard" if variable.lower() in {"pr", "prcp", "precip", "precipitation"} else "standard"


def forward_numpy(values: np.ndarray, spec: TransformSpec) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if spec.kind == "log1p_standard":
        values = np.log1p(np.maximum(values, 0.0))
    elif spec.kind != "standard":
        raise ValueError(f"Unsupported transform {spec.kind!r}")
    return ((values - spec.mean) / spec.std).astype(np.float32, copy=False)


def inverse_numpy(values: np.ndarray, spec: TransformSpec) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32) * spec.std + spec.mean
    if spec.kind == "log1p_standard":
        values = np.maximum(np.expm1(np.minimum(values, LOG1P_MAX)), 0.0)
    elif spec.kind != "standard":
        raise ValueError(f"Unsupported transform {spec.kind!r}")
    return values.astype(np.float32, copy=False)


def inverse_torch(values: torch.Tensor, spec: TransformSpec) -> torch.Tensor:
    values = values * spec.std + spec.mean
    if spec.kind == "log1p_standard":
        return torch.expm1(torch.clamp_max(values, LOG1P_MAX))
    if spec.kind != "standard":
        raise ValueError(f"Unsupported transform {spec.kind!r}")
    return values


def transform_channels_numpy(
    values: np.ndarray,
    variable_names: Iterable[str],
    specs: Mapping[str, TransformSpec],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for channel, name in enumerate(variable_names):
        result[channel] = forward_numpy(values[channel], specs[name])
    return result


def inverse_channels_numpy(
    values: np.ndarray,
    variable_names: Iterable[str],
    specs: Mapping[str, TransformSpec],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for channel, name in enumerate(variable_names):
        result[channel] = inverse_numpy(values[channel], specs[name])
    return result


def specs_from_manifest(manifest: Mapping[str, object]) -> Dict[str, TransformSpec]:
    raw = manifest["transforms"]
    if not isinstance(raw, Mapping):
        raise TypeError("manifest['transforms'] must be a mapping")
    return {str(name): TransformSpec.from_dict(value) for name, value in raw.items()}
