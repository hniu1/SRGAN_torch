"""Streaming preparation of aligned multivariable Daymet/ERA5 arrays."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .transforms import TransformSpec, default_transform_kind


DEFAULT_DATA_ROOT = Path("/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data")
DEFAULT_DEM_ROOT = Path("/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM/final-elev")


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


@dataclass
class StreamingMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, values: np.ndarray, kind: str) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if kind == "log1p_standard":
            values = np.log1p(np.maximum(values, 0.0))
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(np.square(values).sum(dtype=np.float64))

    def finalize(self, name: str, kind: str) -> TransformSpec:
        if self.count == 0:
            raise ValueError(f"No finite training values found for {name}")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return TransformSpec(name=name, kind=kind, mean=mean, std=max(math.sqrt(variance), 1e-6))


def _year_lengths(data_root: Path, variable: str, years: Iterable[int]) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for year in years:
        values = read_variable(source_path(data_root, variable, year, "1deg"), variable)
        result[year] = int(values.shape[0])
    return result


def _atomic_write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _write_shared_time(
    shared_dir: Path,
    split: str,
    years: Sequence[int],
    lengths: Mapping[int, int],
) -> int:
    count = sum(lengths[year] for year in years)
    time_out = np.lib.format.open_memmap(
        shared_dir / f"time_{split}.npy", mode="w+", dtype=np.int16, shape=(count, 2)
    )
    offset = 0
    for year in years:
        n_days = lengths[year]
        time_out[offset:offset + n_days, 0] = year
        time_out[offset:offset + n_days, 1] = np.arange(1, n_days + 1, dtype=np.int16)
        offset += n_days
    time_out.flush()
    return count


def _prepare_shared_data(
    output_dir: Path,
    reference_variable: str,
    split_years: Mapping[str, Sequence[int]],
    data_root: Path,
    dem_root: Path,
    scale_factor: int,
) -> tuple[dict, Dict[int, int]]:
    all_years = sorted({year for years in split_years.values() for year in years})
    lengths = _year_lengths(data_root, reference_variable, all_years)
    first_year = all_years[0]
    lr_first = read_variable(source_path(data_root, reference_variable, first_year, "1deg"), reference_variable)
    hr_first = read_variable(source_path(data_root, reference_variable, first_year, "0p25deg"), reference_variable)
    lr_shape = tuple(int(value) for value in lr_first.shape[-2:])
    hr_shape = tuple(int(value) for value in hr_first.shape[-2:])
    if hr_shape != (lr_shape[0] * scale_factor, lr_shape[1] * scale_factor):
        raise ValueError(f"Expected {scale_factor}x geometry, got LR {lr_shape}, HR {hr_shape}")

    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        split: _write_shared_time(shared_dir, split, tuple(split_years[split]), lengths)
        for split in ("train", "val", "test")
    }
    elevation_lr = read_dem(dem_root / "VICa_DEM_1deg_fill0.nc")
    elevation_hr = read_dem(dem_root / "VICa_DEM_0p25deg_fill0.nc")
    if elevation_lr.shape != lr_shape or elevation_hr.shape != hr_shape:
        raise ValueError(f"DEM geometry mismatch: {elevation_lr.shape}, {elevation_hr.shape}")
    finite_elevation = elevation_hr[np.isfinite(elevation_hr)]
    elevation_mean = float(finite_elevation.mean(dtype=np.float64))
    elevation_std = max(float(finite_elevation.std(dtype=np.float64)), 1e-6)
    np.save(
        shared_dir / "elevation_lr.npy",
        np.nan_to_num(elevation_lr, nan=elevation_mean).astype(np.float32),
    )
    np.save(
        shared_dir / "elevation_hr.npy",
        np.nan_to_num(elevation_hr, nan=elevation_mean).astype(np.float32),
    )
    np.save(shared_dir / "coordinates_lr.npy", normalized_grid_coordinates(*lr_shape))
    np.save(shared_dir / "coordinates_hr.npy", normalized_grid_coordinates(*hr_shape))

    manifest = {
        "format_version": 2,
        "storage_layout": "variable_separable_npy",
        "variables": [],
        "scale_factor": scale_factor,
        "lr_shape": list(lr_shape),
        "hr_shape": list(hr_shape),
        "splits": {
            split: {"years": list(split_years[split]), "samples": split_counts[split]}
            for split in ("train", "val", "test")
        },
        "transforms": {},
        "variable_metadata": {},
        "data_quality": {},
        "static": {"elevation_mean": elevation_mean, "elevation_std": elevation_std},
        "source": {"data_root": str(data_root), "dem_root": str(dem_root)},
    }
    return manifest, lengths


def _validate_existing_manifest(
    manifest: dict,
    split_years: Mapping[str, Sequence[int]],
    scale_factor: int,
    data_root: Path,
    dem_root: Path,
) -> None:
    if manifest.get("format_version") != 2 or manifest.get("storage_layout") != "variable_separable_npy":
        raise ValueError("Prepared data use an older joined layout; choose a new output directory")
    if int(manifest["scale_factor"]) != scale_factor:
        raise ValueError("Existing manifest scale factor does not match the requested scale factor")
    source = manifest.get("source", {})
    if Path(source.get("data_root", "")).resolve() != data_root:
        raise ValueError("Existing manifest data root does not match the requested data root")
    if Path(source.get("dem_root", "")).resolve() != dem_root:
        raise ValueError("Existing manifest DEM root does not match the requested DEM root")
    for split in ("train", "val", "test"):
        if list(manifest["splits"][split]["years"]) != list(split_years[split]):
            raise ValueError(f"Existing {split} years do not match the requested chronological split")


def _shared_year_lengths(output_dir: Path, manifest: dict) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for split in ("train", "val", "test"):
        time = np.load(output_dir / "shared" / f"time_{split}.npy", mmap_mode="r")
        if time.shape[0] != int(manifest["splits"][split]["samples"]):
            raise ValueError(f"Shared {split} time index length does not match manifest")
        years, counts = np.unique(time[:, 0], return_counts=True)
        result.update({int(year): int(count) for year, count in zip(years, counts)})
    return result


def _prepare_variable(
    output_dir: Path,
    variable: str,
    manifest: dict,
    lengths: Mapping[int, int],
    data_root: Path,
) -> tuple[TransformSpec, dict[str, int]]:
    variable_dir = output_dir / "variables" / variable
    variable_dir.mkdir(parents=True, exist_ok=True)
    complete_path = variable_dir / "complete.json"
    complete_path.unlink(missing_ok=True)
    lr_shape = tuple(int(value) for value in manifest["lr_shape"])
    hr_shape = tuple(int(value) for value in manifest["hr_shape"])
    moments = StreamingMoments()
    transform_kind = default_transform_kind(variable)
    valid_lr = np.ones(lr_shape, dtype=bool)
    valid_hr = np.ones(hr_shape, dtype=bool)
    quality = {
        "values_read": 0,
        "missing_values_imputed": 0,
        "negative_precipitation_clamped": 0,
    }

    for split in ("train", "val", "test"):
        years = tuple(int(year) for year in manifest["splits"][split]["years"])
        count = int(manifest["splits"][split]["samples"])
        lr_out = np.lib.format.open_memmap(
            variable_dir / f"lr_{split}.npy", mode="w+", dtype=np.float32,
            shape=(count, *lr_shape),
        )
        hr_out = np.lib.format.open_memmap(
            variable_dir / f"hr_{split}.npy", mode="w+", dtype=np.float32,
            shape=(count, *hr_shape),
        )
        offset = 0
        for year in years:
            print(f"[{variable}/{split}] reading {year}", flush=True)
            lr_read = read_variable_diagnostics(
                source_path(data_root, variable, year, "1deg"), variable
            )
            hr_read = read_variable_diagnostics(
                source_path(data_root, variable, year, "0p25deg"), variable
            )
            lr, hr = lr_read.values, hr_read.values
            if (
                lr_read.canonical_units != hr_read.canonical_units
                or lr_read.unit_conversion != hr_read.unit_conversion
            ):
                raise ValueError(f"{variable} LR/HR unit handling differs for {year}")
            quality["values_read"] += int(lr.size + hr.size)
            quality["missing_values_imputed"] += (
                lr_read.missing_count + hr_read.missing_count
            )
            quality["negative_precipitation_clamped"] += (
                lr_read.negative_clamped_count + hr_read.negative_clamped_count
            )
            expected_days = lengths[year]
            if lr.shape[0] != hr.shape[0] or lr.shape[0] != expected_days:
                raise ValueError(
                    f"{variable} LR/HR time mismatch for {year}: {lr.shape[0]}, {hr.shape[0]}, expected {expected_days}"
                )
            if tuple(lr.shape[-2:]) != lr_shape or tuple(hr.shape[-2:]) != hr_shape:
                raise ValueError(f"{variable} geometry changed in {year}: {lr.shape[-2:]}, {hr.shape[-2:]}")
            days = lr.shape[0]
            lr_out[offset:offset + days] = np.nan_to_num(
                lr, nan=0.0, posinf=0.0, neginf=0.0
            )
            hr_out[offset:offset + days] = np.nan_to_num(
                hr, nan=0.0, posinf=0.0, neginf=0.0
            )
            valid_lr &= np.isfinite(lr).all(axis=0)
            valid_hr &= np.isfinite(hr).all(axis=0)
            if split == "train":
                moments.update(lr, transform_kind)
                moments.update(hr, transform_kind)
            offset += days
            lr_out.flush()
            hr_out.flush()
        if offset != count:
            raise RuntimeError(f"Wrote {offset} {split} samples for {variable}, expected {count}")
    np.save(variable_dir / "valid_lr.npy", valid_lr.astype(np.float32))
    np.save(variable_dir / "valid_hr.npy", valid_hr.astype(np.float32))
    spec = moments.finalize(variable, transform_kind)
    _atomic_write_manifest(
        complete_path,
        {
            "variable": variable,
            "lr_shape": list(lr_shape),
            "hr_shape": list(hr_shape),
            "splits": {
                split: int(manifest["splits"][split]["samples"])
                for split in ("train", "val", "test")
            },
        },
    )
    return spec, quality


def prepare_dataset(
    output_dir: Path,
    variables: Sequence[str],
    split_years: Mapping[str, Sequence[int]],
    data_root: Path = DEFAULT_DATA_ROOT,
    dem_root: Path = DEFAULT_DEM_ROOT,
    scale_factor: int = 4,
    overwrite: bool = False,
) -> dict:
    """Prepare appendable per-variable arrays and shared static/time metadata."""
    output_dir = output_dir.resolve()
    data_root = Path(data_root).resolve()
    dem_root = Path(dem_root).resolve()
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = tuple(name.lower() for name in variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be a non-empty sequence without duplicates")
    for required in ("train", "val", "test"):
        if required not in split_years or not split_years[required]:
            raise ValueError(f"split_years must contain a non-empty {required!r} split")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _validate_existing_manifest(manifest, split_years, scale_factor, data_root, dem_root)
        lengths = _shared_year_lengths(output_dir, manifest)
    else:
        manifest, lengths = _prepare_shared_data(
            output_dir, variables[0], split_years, data_root, dem_root, scale_factor
        )
        _atomic_write_manifest(manifest_path, manifest)

    first_year = min(year for years in split_years.values() for year in years)
    for variable in variables:
        if variable in manifest["variables"] and not overwrite:
            complete = output_dir / "variables" / variable / "complete.json"
            if not complete.exists():
                raise RuntimeError(
                    f"{variable} is registered but incomplete; rerun with --overwrite"
                )
            print(f"[{variable}] already prepared; skipping (use --overwrite to replace it)", flush=True)
            continue
        spec, quality = _prepare_variable(
            output_dir, variable, manifest, lengths, data_root
        )
        if variable not in manifest["variables"]:
            manifest["variables"].append(variable)
        manifest["transforms"][variable] = spec.to_dict()
        manifest.setdefault("data_quality", {})[variable] = quality
        manifest["variable_metadata"][variable] = read_variable_metadata(
            source_path(data_root, variable, first_year, "1deg"), variable
        )
        _atomic_write_manifest(manifest_path, manifest)
        print(f"[{variable}] prepared and registered", flush=True)

    print(f"Prepared dataset: {manifest_path}", flush=True)
    return manifest
