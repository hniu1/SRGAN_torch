"""Index aligned TVA Daymet grids for the two Genesis 5x REFINE stages."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .prepare import _netcdf_dataset, normalized_grid_coordinates, read_variable_metadata
from .transforms import TransformSpec, default_transform_kind


DEFAULT_TVA_ROOT = Path("/lustre/orion/proj-shared/cli138/dr6/Daymet1km/daymet_TVA_latlon")
STAGE_RESOLUTIONS = {1: ("025deg", "005deg"), 2: ("005deg", "001deg")}
RESOLUTION_DEGREES = {"025deg": 0.25, "005deg": 0.05, "001deg": 0.01}


def daymet_source_path(root: Path, variable: str, year: int, resolution: str) -> Path:
    return root / resolution / variable / f"daymet_TVA_{variable}_{year}_{resolution}.nc"


def read_grid(dataset) -> tuple[np.ndarray, np.ndarray]:
    coordinates = tuple(np.asarray(dataset.variables[name][:], dtype=np.float64) for name in ("lat", "lon"))
    if any(v.ndim != 1 or not np.isfinite(v).all() or len(v) < 2 for v in coordinates):
        raise ValueError(f"Expected finite one-dimensional lat/lon coordinates: {dataset.filepath()}")
    return coordinates


def check_grid(actual, expected, description: str) -> None:
    for name, values, reference in zip(("lat", "lon"), actual, expected):
        if values.shape != reference.shape or not np.allclose(values, reference, rtol=0, atol=1e-6):
            raise ValueError(f"{description}: {name} coordinates do not match the training grid")


def read_dates(dataset) -> list[dt.datetime]:
    from netCDF4 import num2date

    time = dataset.variables["time"]
    calendar = getattr(time, "calendar", "standard")
    if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise ValueError(f"Unsupported Daymet calendar: {calendar}")
    values = np.asarray(time[:])
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Time coordinates must be finite and strictly increasing")
    dates = num2date(values, time.units, calendar)
    return [dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in dates]


def _clean(values, metadata: dict, variable: str) -> np.ndarray:
    values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)
    valid = np.isfinite(values)
    if metadata["unit_conversion"] == "kelvin_to_celsius":
        values = values - np.float32(273.15)
    if default_transform_kind(variable) == "log1p_standard":
        values = np.maximum(values, 0)
    values[~valid] = np.nan
    return values


def fit_transforms(root: Path, variables: Sequence[str], years: Sequence[int], resolution: str,
                   metadata: dict, chunk_days: int = 8) -> dict:
    """Fit the existing standard/log1p transforms on training LR fields only."""
    result = {}
    for variable in variables:
        count, mean, m2 = 0, 0.0, 0.0
        kind = default_transform_kind(variable)
        for year in years:
            with _netcdf_dataset(daymet_source_path(root, variable, year, resolution)) as dataset:
                field = dataset.variables[variable]
                for start in range(0, field.shape[0], chunk_days):
                    values = _clean(field[start:start + chunk_days], metadata[variable], variable)
                    values = values[np.isfinite(values)].astype(np.float64)
                    if kind == "log1p_standard":
                        values = np.log1p(values)
                    n = values.size
                    if not n:
                        continue
                    batch_mean = float(values.mean())
                    batch_m2 = float(np.square(values - batch_mean).sum())
                    delta = batch_mean - mean
                    total = count + n
                    m2 += batch_m2 + delta * delta * count * n / total
                    mean += delta * n / total
                    count = total
        if not count:
            raise ValueError(f"No finite training values for {variable}")
        result[variable] = TransformSpec(variable, kind, mean, max(float(np.sqrt(m2 / count)), 1e-6)).to_dict()
        print(f"Fitted {variable} normalization on {count} training values", flush=True)
    return result


def _read_terrain(path: Path, grid) -> np.ndarray:
    with _netcdf_dataset(path) as dataset:
        check_grid(read_grid(dataset), grid, f"DEM {path}")
        names = [name for name in ("DEM", "dem", "elevation", "elev") if name in dataset.variables]
        if not names:
            raise ValueError(f"No DEM/dem/elevation/elev variable in {path}")
        field = dataset.variables[names[0]]
        if field.dimensions[-2:] != ("lat", "lon"):
            raise ValueError(f"DEM dimensions must end in lat,lon: {path}")
        elevation = np.asarray(np.ma.filled(field[:], np.nan), dtype=np.float32).squeeze()
    if elevation.shape != tuple(len(v) for v in grid) or not np.isfinite(elevation).any():
        raise ValueError(f"DEM has invalid geometry or no finite elevation: {path}")
    return elevation


def prepare_daymet_index(output_dir: Path, variables: Sequence[str], split_years: Mapping[str, Sequence[int]],
                         data_root: Path = DEFAULT_TVA_ROOT, stage: int = 1,
                         dem_lr: Path | None = None, dem_hr: Path | None = None,
                         normalization_manifest_path: Path | None = None) -> dict:
    if stage not in STAGE_RESOLUTIONS:
        raise ValueError("stage must be 1 (025deg -> 005deg) or 2 (005deg -> 001deg)")
    if dem_lr is None or dem_hr is None:
        raise ValueError("Both --dem-lr and --dem-hr are required to preserve terrain-aware REFINE")
    output_dir, data_root = Path(output_dir).resolve(), Path(data_root).resolve()
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Prepared manifest already exists: {output_dir / 'manifest.json'}")
    variables = tuple(str(v).lower() for v in variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("Variables must be nonempty and unique")
    splits = {name: [int(y) for y in split_years[name]] for name in ("train", "val", "test")}
    if any(not years or len(years) != len(set(years)) or years != sorted(years) for years in splits.values()):
        raise ValueError("Each split must contain sorted, unique, nonempty years")
    if not (max(splits["train"]) < min(splits["val"]) and max(splits["val"]) < min(splits["test"])):
        raise ValueError("Train, validation and test years must be chronologically disjoint")
    lr_res, hr_res = STAGE_RESOLUTIONS[stage]
    grids, dates_by_year, metadata, domain_masks = {}, {}, {}, {}
    for variable in variables:
        for year in sorted(y for years in splits.values() for y in years):
            for resolution in (lr_res, hr_res):
                path = daymet_source_path(data_root, variable, year, resolution)
                with _netcdf_dataset(path) as dataset:
                    field = dataset.variables[variable]
                    if field.dimensions != ("time", "lat", "lon"):
                        raise ValueError(f"Expected time,lat,lon dimensions: {path}")
                    grid = read_grid(dataset)
                    if resolution in grids:
                        check_grid(grid, grids[resolution], str(path))
                    else:
                        grids[resolution] = grid
                    spacing = RESOLUTION_DEGREES[resolution]
                    if any(not np.allclose(np.diff(axis), spacing, rtol=0, atol=1e-6) for axis in grid):
                        raise ValueError(f"Unexpected coordinate spacing/orientation: {path}")
                    dates = read_dates(dataset)
                    if len(dates) != field.shape[0] or any(d.year != year for d in dates):
                        raise ValueError(f"Time coordinates do not match source year: {path}")
                    if year in dates_by_year and dates != dates_by_year[year]:
                        raise ValueError(f"Time coordinates differ across variables/resolutions: {path}")
                    dates_by_year[year] = dates
                    if year == splits["train"][0]:
                        valid = np.isfinite(np.ma.filled(field[0], np.nan))
                        domain_masks[resolution] = domain_masks.get(resolution, np.ones_like(valid)) & valid
                current_metadata = read_variable_metadata(path, variable)
                if variable in metadata and current_metadata != metadata[variable]:
                    raise ValueError(f"Variable metadata changed across years/resolutions: {path}")
                metadata[variable] = current_metadata

    for coarse, fine in zip(grids[lr_res], grids[hr_res]):
        if len(fine) != len(coarse) * 5 or not np.allclose(fine.reshape(-1, 5).mean(axis=1), coarse, rtol=0, atol=1e-6):
            raise ValueError("LR/HR grids must have aligned 5x cell boundaries")
    elevation_lr = _read_terrain(Path(dem_lr), grids[lr_res])
    elevation_hr = _read_terrain(Path(dem_hr), grids[hr_res])
    if normalization_manifest_path is None:
        transforms = fit_transforms(data_root, variables, splits["train"], lr_res, metadata)
    else:
        normalization = json.loads(Path(normalization_manifest_path).read_text())
        if normalization.get("normalization_fit") != {"years": splits["train"], "resolution": lr_res}:
            raise ValueError("External normalization must document the same training years and LR resolution")
        transforms = {v: normalization["transforms"][v] for v in variables}
    for variable, raw in transforms.items():
        spec = TransformSpec.from_dict(raw)
        if spec.name != variable or spec.kind != default_transform_kind(variable) or not np.isfinite([spec.mean, spec.std]).all() or spec.std <= 0:
            raise ValueError(f"Invalid normalization transform for {variable}")

    shared = output_dir / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    finite = elevation_hr[np.isfinite(elevation_hr)]
    elevation_mean, elevation_std = float(finite.mean(dtype=np.float64)), max(float(finite.std(dtype=np.float64)), 1e-6)
    for side, resolution, elevation in (("lr", lr_res, elevation_lr), ("hr", hr_res, elevation_hr)):
        np.save(shared / f"elevation_{side}.npy", np.nan_to_num(elevation, nan=elevation_mean))
        np.save(shared / f"coordinates_{side}.npy", normalized_grid_coordinates(*elevation.shape))
        np.save(shared / f"valid_{side}.npy", (np.isfinite(elevation) & domain_masks[resolution]).astype(np.float32))
        for name, values in zip(("lat", "lon"), grids[resolution]):
            np.save(shared / f"{name}_{side}.npy", values)
    records = {}
    for split, years in splits.items():
        dates = [d for y in years for d in dates_by_year[y]]
        np.save(shared / f"time_{split}.npy", np.asarray([(d.year, d.timetuple().tm_yday) for d in dates], dtype=np.int16))
        np.save(shared / f"file_index_{split}.npy", np.asarray([i for y in years for i in range(len(dates_by_year[y]))], dtype=np.int32))
        records[split] = {"years": years, "samples": len(dates)}
    manifest = {
        "format_version": 3, "storage_layout": "netcdf_patch_index", "stage": stage,
        "variables": list(variables), "scale_factor": 5,
        "lr_resolution_degrees": RESOLUTION_DEGREES[lr_res], "hr_resolution_degrees": RESOLUTION_DEGREES[hr_res],
        "lr_shape": list(elevation_lr.shape), "hr_shape": list(elevation_hr.shape),
        "grid_bounds": {resolution: [[float(axis[0]), float(axis[-1])] for axis in grids[resolution]]
                        for resolution in (lr_res, hr_res)},
        "splits": records, "year_lengths": {str(y): len(d) for y, d in dates_by_year.items()},
        "transforms": transforms, "variable_metadata": metadata,
        "normalization_fit": {"years": splits["train"], "resolution": lr_res},
        "static": {"elevation_mean": elevation_mean, "elevation_std": elevation_std},
        "source": {"layout": "daymet_tva", "data_root": os.path.relpath(data_root, output_dir),
                   "lr_suffix": lr_res, "hr_suffix": hr_res,
                   "dem_lr": str(Path(dem_lr).resolve()), "dem_hr": str(Path(dem_hr).resolve())},
        "data_quality": {"policy": "masked/nonfinite values are zero-filled after normalization and excluded per patch",
                         "negative_precipitation": "clamped to zero on read",
                         "time": "source CF timestamps retained; file indexes stored separately from day of year",
                         "domain_mask": "finite DEM and all climate variables on the first training timestep"},
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output_dir / "manifest.json")
    print(f"Prepared Stage {stage} 5x index: {output_dir / 'manifest.json'}", flush=True)
    return manifest
