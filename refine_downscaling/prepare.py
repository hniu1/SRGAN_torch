"""Streaming preparation of aligned multivariable Daymet/ERA5 arrays."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .transforms import TransformSpec, default_transform_kind


DEFAULT_DATA_ROOT = Path("daymet/data")
DEFAULT_DEM_ROOT = Path("daymet/dem")


def _netcdf_dataset(path: Path):
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required for data preparation") from exc
    return Dataset(path)


def source_path(data_root: Path, variable: str, year: int, resolution: str) -> Path:
    return data_root / f"Daymet_ERA5_{variable}_dy_{year}_{resolution}.nc"


def _variable_family(variable: str) -> str:
    name = variable.lower()
    if name in {"tmin", "tmax", "tasmin", "tasmax", "temperature"}:
        return "temperature"
    if name in {"pr", "prcp", "precip", "precipitation"}:
        return "precipitation"
    return "other"


def _canonical_units(variable: str, source_units: str) -> tuple[str, str]:
    family = _variable_family(variable)
    compact = source_units.strip().lower().replace("°", "deg").replace(" ", "")
    if family == "temperature":
        if compact in {"c", "degc", "degreec", "degreesc", "celsius"}:
            return "degC", "none"
        if compact in {"k", "degk", "kelvin"}:
            return "degC", "kelvin_to_celsius"
        raise ValueError(f"Unsupported temperature units {source_units!r} for {variable}")
    if family == "precipitation":
        if compact in {"mm/dy", "mm/day", "mmd-1", "mmday-1"}:
            return "mm/day", "none"
        raise ValueError(f"Unsupported daily precipitation units {source_units!r} for {variable}")
    return source_units, "none"


@dataclass(frozen=True)
class VariableRead:
    values: np.ndarray
    source_units: str
    canonical_units: str
    unit_conversion: str
    missing_count: int
    negative_clamped_count: int


def read_variable_diagnostics(path: Path, variable: str) -> VariableRead:
    if not path.exists():
        raise FileNotFoundError(path)
    with _netcdf_dataset(path) as dataset:
        key = f"{variable}_dy"
        if key not in dataset.variables:
            raise KeyError(f"{key!r} is missing from {path}")
        field = dataset.variables[key]
        source_units = str(getattr(field, "units", ""))
        canonical_units, conversion = _canonical_units(variable, source_units)
        masked = np.ma.asarray(field[:])
        values = np.ma.filled(masked, np.nan)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [time, y, x] in {path}, got {values.shape}")
    finite = np.isfinite(values)
    missing_count = int((~finite).sum())
    if conversion == "kelvin_to_celsius":
        values[finite] -= np.float32(273.15)
    negative_clamped_count = 0
    if _variable_family(variable) == "precipitation":
        negative = finite & (values < 0.0)
        negative_clamped_count = int(negative.sum())
        values[negative] = 0.0
    return VariableRead(
        values=values,
        source_units=source_units,
        canonical_units=canonical_units,
        unit_conversion=conversion,
        missing_count=missing_count,
        negative_clamped_count=negative_clamped_count,
    )


def read_variable(path: Path, variable: str) -> np.ndarray:
    return read_variable_diagnostics(path, variable).values


def read_variable_metadata(path: Path, variable: str) -> dict[str, str]:
    with _netcdf_dataset(path) as dataset:
        field = dataset.variables[f"{variable}_dy"]
        source_units = str(getattr(field, "units", ""))
        canonical_units, conversion = _canonical_units(variable, source_units)
        return {
            "source_units": source_units,
            "units": canonical_units,
            "unit_conversion": conversion,
            "long_name": str(getattr(field, "long_name", variable)),
        }


def read_dem(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with _netcdf_dataset(path) as dataset:
        if "DEM" not in dataset.variables:
            raise KeyError(f"'DEM' is missing from {path}")
        values = np.ma.filled(dataset.variables["DEM"][:], np.nan)
    values = np.asarray(values, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional DEM in {path}, got {values.shape}")
    return values


def normalized_grid_coordinates(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([yy, xx], axis=0)
