"""Create a lightweight index for lazy 0.25-degree to 1/24-degree patches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .prepare import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DEM_ROOT,
    normalized_grid_coordinates,
    read_dem,
    read_variable_metadata,
)


LR_SUFFIX = "0p25deg"
HR_SUFFIX = "trim"


def stage2_source_path(data_root: Path, variable: str, year: int, suffix: str) -> Path:
    return data_root / f"Daymet_ERA5_{variable}_dy_{year}_{suffix}.nc"


def _field_shape(path: Path, variable: str) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required for Stage-2 preparation") from exc
    with Dataset(path) as dataset:
        key = f"{variable}_dy"
        if key not in dataset.variables:
            raise KeyError(f"{key!r} is missing from {path}")
        shape = tuple(int(value) for value in dataset.variables[key].shape)
    if len(shape) != 3:
        raise ValueError(f"Expected [time, y, x] in {path}, got {shape}")
    return shape


def prepare_stage2_index(
    output_dir: Path,
    variables: Sequence[str],
    split_years: Mapping[str, Sequence[int]],
    stage1_manifest_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    dem_root: Path = DEFAULT_DEM_ROOT,
    scale_factor: int = 6,
) -> dict:
    """Validate all source pairs and write only metadata/static fields, not 200 GB of copies."""
    if scale_factor != 6:
        raise ValueError("The current Stage-2 source grids require scale_factor=6")
    output_dir = Path(output_dir).resolve()
    data_root = Path(data_root).resolve()
    dem_root = Path(dem_root).resolve()
    stage1_manifest_path = Path(stage1_manifest_path).resolve()
    variables = tuple(str(name).lower() for name in variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be non-empty and unique")
    for split in ("train", "val", "test"):
        if split not in split_years or not split_years[split]:
            raise ValueError(f"Missing non-empty {split} years")
    if not stage1_manifest_path.exists():
        raise FileNotFoundError(stage1_manifest_path)
    stage1_manifest = json.loads(stage1_manifest_path.read_text())
    missing_transforms = set(variables) - set(stage1_manifest.get("transforms", {}))
    if missing_transforms:
        raise ValueError(f"Stage-1 transforms are missing {sorted(missing_transforms)}")

    all_years = sorted({int(year) for years in split_years.values() for year in years})
    year_lengths: dict[str, int] = {}
    lr_shape: tuple[int, int] | None = None
    hr_shape: tuple[int, int] | None = None
    variable_metadata: dict[str, dict] = {}
    for variable in variables:
        for year in all_years:
            lr_path = stage2_source_path(data_root, variable, year, LR_SUFFIX)
            hr_path = stage2_source_path(data_root, variable, year, HR_SUFFIX)
            lr_field = _field_shape(lr_path, variable)
            hr_field = _field_shape(hr_path, variable)
            if lr_field[0] != hr_field[0]:
                raise ValueError(f"{variable} time mismatch for {year}: {lr_field}, {hr_field}")
            year_key = str(year)
            if year_key in year_lengths and year_lengths[year_key] != lr_field[0]:
                raise ValueError(f"Variable time mismatch for {year}")
            year_lengths[year_key] = lr_field[0]
            current_lr = lr_field[-2:]
            current_hr = hr_field[-2:]
            if current_hr != tuple(value * scale_factor for value in current_lr):
                raise ValueError(f"Expected 6x geometry for {variable}/{year}: {current_lr}, {current_hr}")
            if lr_shape is not None and (current_lr != lr_shape or current_hr != hr_shape):
                raise ValueError(f"Grid geometry changed for {variable}/{year}")
            lr_shape, hr_shape = current_lr, current_hr
        lr_metadata = read_variable_metadata(
            stage2_source_path(data_root, variable, all_years[0], LR_SUFFIX), variable
        )
        hr_metadata = read_variable_metadata(
            stage2_source_path(data_root, variable, all_years[0], HR_SUFFIX), variable
        )
        if (
            lr_metadata["units"] != hr_metadata["units"]
            or lr_metadata["unit_conversion"] != hr_metadata["unit_conversion"]
        ):
            raise ValueError(f"{variable} LR/HR unit handling differs")
        variable_metadata[variable] = lr_metadata

    assert lr_shape is not None and hr_shape is not None
    elevation_lr = read_dem(dem_root / "VICa_DEM_0p25deg_fill0.nc")
    elevation_hr = read_dem(dem_root / "VICa_DEM_trim_fill0.nc")
    if elevation_lr.shape != lr_shape or elevation_hr.shape != hr_shape:
        raise ValueError(
            f"Stage-2 DEM geometry mismatch: {elevation_lr.shape}, {elevation_hr.shape}"
        )
    finite_hr = elevation_hr[np.isfinite(elevation_hr)]
    elevation_mean = float(finite_hr.mean(dtype=np.float64))
    elevation_std = max(float(finite_hr.std(dtype=np.float64)), 1e-6)

    shared = output_dir / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    np.save(shared / "elevation_lr.npy", np.nan_to_num(elevation_lr, nan=elevation_mean).astype(np.float32))
    np.save(shared / "elevation_hr.npy", np.nan_to_num(elevation_hr, nan=elevation_mean).astype(np.float32))
    np.save(shared / "coordinates_lr.npy", normalized_grid_coordinates(*lr_shape))
    np.save(shared / "coordinates_hr.npy", normalized_grid_coordinates(*hr_shape))
    np.save(shared / "valid_lr.npy", np.isfinite(elevation_lr).astype(np.float32))
    np.save(shared / "valid_hr.npy", np.isfinite(elevation_hr).astype(np.float32))

    split_records = {}
    for split in ("train", "val", "test"):
        years = [int(year) for year in split_years[split]]
        samples = sum(year_lengths[str(year)] for year in years)
        split_records[split] = {"years": years, "samples": samples}
        time = np.empty((samples, 2), dtype=np.int16)
        offset = 0
        for year in years:
            days = year_lengths[str(year)]
            time[offset:offset + days, 0] = year
            time[offset:offset + days, 1] = np.arange(1, days + 1, dtype=np.int16)
            offset += days
        np.save(shared / f"time_{split}.npy", time)

    manifest = {
        "format_version": 3,
        "storage_layout": "netcdf_patch_index",
        "stage": 2,
        "variables": list(variables),
        "scale_factor": scale_factor,
        "lr_resolution_degrees": 0.25,
        "hr_resolution_degrees": 1.0 / 24.0,
        "lr_shape": list(lr_shape),
        "hr_shape": list(hr_shape),
        "splits": split_records,
        "year_lengths": year_lengths,
        "transforms": {name: stage1_manifest["transforms"][name] for name in variables},
        "variable_metadata": variable_metadata,
        "static": {"elevation_mean": elevation_mean, "elevation_std": elevation_std},
        "source": {
            "data_root": str(data_root),
            "lr_suffix": LR_SUFFIX,
            "hr_suffix": HR_SUFFIX,
            "dem_root": str(dem_root),
            "stage1_manifest": str(stage1_manifest_path),
        },
        "data_quality": {
            "policy": "masked/nonfinite values are zero-filled after normalization and excluded per patch",
            "negative_precipitation": "clamped to zero on read",
        },
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output_dir / "manifest.json")
    print(f"Prepared Stage-2 lazy index: {output_dir / 'manifest.json'}", flush=True)
    return manifest
