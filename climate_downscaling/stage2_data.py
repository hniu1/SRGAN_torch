"""Lazy NetCDF patch/full-field datasets for 0.25-degree to 1/24-degree Stage 2."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import _crop_with_reflection, _validation_origins
from .stage2_prepare import stage2_source_path
from .transforms import TransformSpec, specs_from_manifest, transform_channels_numpy


def load_stage2_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 3 or manifest.get("storage_layout") != "netcdf_patch_index":
        raise ValueError(f"Unsupported Stage-2 manifest: {path}")
    return manifest


def _select_variables(manifest: dict, requested: Sequence[str] | None) -> tuple[str, ...]:
    available = tuple(str(name) for name in manifest["variables"])
    selected = available if requested is None else tuple(str(name) for name in requested)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Stage-2 variables must be non-empty and unique")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Stage-2 variables are not indexed: {sorted(missing)}")
    return selected


def _direct_crop(field, day: int, row: int, col: int, height: int, width: int) -> np.ndarray:
    source_h, source_w = field.shape[-2:]
    top = max(-row, 0)
    left = max(-col, 0)
    bottom = max(row + height - source_h, 0)
    right = max(col + width - source_w, 0)
    row0, row1 = max(row, 0), min(row + height, source_h)
    col0, col1 = max(col, 0), min(col + width, source_w)
    values = np.asarray(np.ma.filled(field[day, row0:row1, col0:col1], np.nan), dtype=np.float32)
    if top or bottom or left or right:
        values = np.pad(values, ((top, bottom), (left, right)), mode="reflect")
    if values.shape != (height, width):
        raise RuntimeError(f"NetCDF crop produced {values.shape}, expected {(height, width)}")
    return values


class _Stage2SourceMixin:
    def _initialize_source(self, data_dir: Path, split: str, variable_names: Sequence[str] | None) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = load_stage2_manifest(self.data_dir)
        self.variable_names = _select_variables(self.manifest, variable_names)
        self.specs: dict[str, TransformSpec] = specs_from_manifest(self.manifest)
        self.split = split
        self.scale_factor = int(self.manifest["scale_factor"])
        self.lr_shape = tuple(int(value) for value in self.manifest["lr_shape"])
        self.hr_shape = tuple(int(value) for value in self.manifest["hr_shape"])
        self.time = np.load(self.data_dir / "shared" / f"time_{split}.npy", mmap_mode="r")
        self.data_root = Path(self.manifest["source"]["data_root"])
        self.lr_suffix = str(self.manifest["source"]["lr_suffix"])
        self.hr_suffix = str(self.manifest["source"]["hr_suffix"])
        self._handles: OrderedDict[str, object] = OrderedDict()
        self.max_open_files = 64

    def _dataset(self, path: Path):
        key = str(path)
        if key in self._handles:
            dataset = self._handles.pop(key)
            self._handles[key] = dataset
            return dataset
        try:
            from netCDF4 import Dataset as NetCDFDataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required for Stage-2 patch reads") from exc
        dataset = NetCDFDataset(path)
        self._handles[key] = dataset
        while len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        return dataset

    def _field(self, variable: str, year: int, suffix: str):
        path = stage2_source_path(self.data_root, variable, year, suffix)
        return self._dataset(path).variables[f"{variable}_dy"]

    def _clean(self, values: np.ndarray, variable: str) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(values)
        metadata = self.manifest.get("variable_metadata", {}).get(variable, {})
        if metadata.get("unit_conversion") == "kelvin_to_celsius":
            values = values.copy()
            values[valid] -= np.float32(273.15)
        if variable in {"pr", "prcp", "precip", "precipitation"}:
            values = values.copy()
            values[valid & (values < 0.0)] = 0.0
        return values, valid

    def _read_patch(
        self, variable: str, year: int, day: int, suffix: str,
        row: int, col: int, height: int, width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _direct_crop(self._field(variable, year, suffix), day, row, col, height, width)
        return self._clean(values, variable)

    def _read_full(self, variable: str, year: int, day: int, suffix: str) -> tuple[np.ndarray, np.ndarray]:
        field = self._field(variable, year, suffix)
        values = np.asarray(np.ma.filled(field[day], np.nan), dtype=np.float32)
        return self._clean(values, variable)

    def close(self) -> None:
        for dataset in self._handles.values():
            dataset.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Stage2NetCDFPatchDataset(_Stage2SourceMixin, Dataset):
    """Read only the aligned LR/HR crop requested for one training sample."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        core_size: int = 8,
        halo: int = 2,
        patches_per_day: int = 8,
        random_patches: bool = True,
        variable_names: Sequence[str] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self._initialize_source(data_dir, split, variable_names)
        self.core_size = int(core_size)
        self.halo = int(halo)
        self.context_size = self.core_size + 2 * self.halo
        self.patches_per_day = int(patches_per_day)
        self.random_patches = bool(random_patches)
        if self.core_size <= 0 or self.halo < 0 or self.patches_per_day <= 0:
            raise ValueError("Invalid Stage-2 patch geometry")
        if self.core_size > min(self.lr_shape):
            raise ValueError("Stage-2 core exceeds the LR field")

        shared = self.data_dir / "shared"
        self.elevation_lr = np.load(shared / "elevation_lr.npy", mmap_mode="r")
        self.elevation_hr = np.load(shared / "elevation_hr.npy", mmap_mode="r")
        self.coordinates_lr = np.load(shared / "coordinates_lr.npy", mmap_mode="r")
        self.coordinates_hr = np.load(shared / "coordinates_hr.npy", mmap_mode="r")
        self.domain_valid_lr = np.load(shared / "valid_lr.npy", mmap_mode="r")
        self.domain_valid_hr = np.load(shared / "valid_hr.npy", mmap_mode="r")
        self.elevation_mean = float(self.manifest["static"]["elevation_mean"])
        self.elevation_std = float(self.manifest["static"]["elevation_std"])
        self._deterministic_origins = _validation_origins(
            self.lr_shape[0], self.lr_shape[1], self.core_size, self.patches_per_day
        )

    def __len__(self) -> int:
        return int(self.time.shape[0]) * self.patches_per_day

    def _origin(self, patch_index: int) -> tuple[int, int]:
        if not self.random_patches:
            return self._deterministic_origins[patch_index]
        return (
            int(np.random.randint(0, self.lr_shape[0] - self.core_size + 1)),
            int(np.random.randint(0, self.lr_shape[1] - self.core_size + 1)),
        )

    def _static_patch(
        self, row: int, col: int, valid_lr: np.ndarray, valid_hr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.context_size
        factor = self.scale_factor
        hr_size = size * factor
        lr_elevation = _crop_with_reflection(self.elevation_lr, row, col, size, size)[None]
        lr_coordinates = _crop_with_reflection(self.coordinates_lr, row, col, size, size)
        domain_lr = _crop_with_reflection(self.domain_valid_lr, row, col, size, size) > 0.5
        hr_elevation = _crop_with_reflection(
            self.elevation_hr, row * factor, col * factor, hr_size, hr_size
        )[None]
        hr_coordinates = _crop_with_reflection(
            self.coordinates_hr, row * factor, col * factor, hr_size, hr_size
        )
        domain_hr = _crop_with_reflection(
            self.domain_valid_hr, row * factor, col * factor, hr_size, hr_size
        ) > 0.5
        lr_elevation = (lr_elevation.astype(np.float32) - self.elevation_mean) / self.elevation_std
        hr_elevation = (hr_elevation.astype(np.float32) - self.elevation_mean) / self.elevation_std
        static_lr = np.concatenate([
            lr_elevation, lr_coordinates.astype(np.float32), (valid_lr & domain_lr)[None].astype(np.float32)
        ])
        static_hr = np.concatenate([
            hr_elevation, hr_coordinates.astype(np.float32), (valid_hr & domain_hr)[None].astype(np.float32)
        ])
        return static_lr, static_hr

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = index // self.patches_per_day
        patch_index = index % self.patches_per_day
        year, day_of_year = (int(value) for value in self.time[sample])
        day = day_of_year - 1
        core_row, core_col = self._origin(patch_index)
        row, col = core_row - self.halo, core_col - self.halo
        size = self.context_size
        factor = self.scale_factor
        hr_size = size * factor

        lr_values, lr_valid_fields = [], []
        hr_values, hr_valid_fields = [], []
        for variable in self.variable_names:
            values, valid = self._read_patch(variable, year, day, self.lr_suffix, row, col, size, size)
            lr_values.append(values)
            lr_valid_fields.append(valid)
            values, valid = self._read_patch(
                variable, year, day, self.hr_suffix,
                row * factor, col * factor, hr_size, hr_size,
            )
            hr_values.append(values)
            hr_valid_fields.append(valid)
        lr_raw = np.stack(lr_values)
        hr_raw = np.stack(hr_values)
        valid_lr = np.logical_and.reduce(lr_valid_fields)
        valid_hr = np.logical_and.reduce(hr_valid_fields)
        lr_scaled = transform_channels_numpy(
            np.nan_to_num(lr_raw, nan=0.0, posinf=0.0, neginf=0.0),
            self.variable_names, self.specs,
        )
        hr_scaled = transform_channels_numpy(
            np.nan_to_num(hr_raw, nan=0.0, posinf=0.0, neginf=0.0),
            self.variable_names, self.specs,
        )
        lr_scaled[:, ~valid_lr] = 0.0
        hr_scaled[:, ~valid_hr] = 0.0
        static_lr, static_hr = self._static_patch(row, col, valid_lr, valid_hr)

        loss_mask = np.zeros((1, hr_size, hr_size), dtype=np.float32)
        margin = self.halo * factor
        core_hr = self.core_size * factor
        loss_mask[:, margin:margin + core_hr, margin:margin + core_hr] = 1.0
        loss_mask *= static_hr[-1:]
        phase = 2.0 * math.pi * (day_of_year - 1.0) / 365.25
        return {
            "lr": torch.from_numpy(lr_scaled.copy()),
            "target": torch.from_numpy(hr_scaled.copy()),
            "static_lr": torch.from_numpy(static_lr.copy()),
            "static_hr": torch.from_numpy(static_hr.copy()),
            "loss_mask": torch.from_numpy(loss_mask),
            "season": torch.tensor([math.sin(phase), math.cos(phase)], dtype=torch.float32),
        }


class Stage2FullFieldDataset(_Stage2SourceMixin, Dataset):
    """Lazily read one complete 0.25-degree/1/24-degree day for evaluation."""

    def __init__(
        self, data_dir: Path, split: str = "test", variable_names: Sequence[str] | None = None
    ) -> None:
        self._initialize_source(data_dir, split, variable_names)
        shared = self.data_dir / "shared"
        elevation_mean = float(self.manifest["static"]["elevation_mean"])
        elevation_std = float(self.manifest["static"]["elevation_std"])
        elevation_lr = np.load(shared / "elevation_lr.npy").astype(np.float32)
        elevation_hr = np.load(shared / "elevation_hr.npy").astype(np.float32)
        coordinates_lr = np.load(shared / "coordinates_lr.npy").astype(np.float32)
        coordinates_hr = np.load(shared / "coordinates_hr.npy").astype(np.float32)
        valid_lr = np.load(shared / "valid_lr.npy").astype(np.float32)
        valid_hr = np.load(shared / "valid_hr.npy").astype(np.float32)
        self.static_lr = np.concatenate([
            ((elevation_lr - elevation_mean) / elevation_std)[None], coordinates_lr, valid_lr[None]
        ]).astype(np.float32)
        self.static_hr = np.concatenate([
            ((elevation_hr - elevation_mean) / elevation_std)[None], coordinates_hr, valid_hr[None]
        ]).astype(np.float32)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        year, day_of_year = (int(value) for value in self.time[index])
        day = day_of_year - 1
        lr_values, lr_valid_fields = [], []
        hr_values, hr_valid_fields = [], []
        for variable in self.variable_names:
            values, valid = self._read_full(variable, year, day, self.lr_suffix)
            lr_values.append(values)
            lr_valid_fields.append(valid)
            values, valid = self._read_full(variable, year, day, self.hr_suffix)
            hr_values.append(values)
            hr_valid_fields.append(valid)
        lr_raw = np.stack(lr_values)
        target_raw = np.stack(hr_values)
        valid_lr = np.logical_and.reduce(lr_valid_fields)
        valid_hr = np.logical_and.reduce(hr_valid_fields)
        lr_scaled = transform_channels_numpy(
            np.nan_to_num(lr_raw, nan=0.0, posinf=0.0, neginf=0.0),
            self.variable_names, self.specs,
        )
        lr_scaled[:, ~valid_lr] = 0.0
        target_raw = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)
        static_lr = self.static_lr.copy()
        static_hr = self.static_hr.copy()
        static_lr[-1] *= valid_lr
        static_hr[-1] *= valid_hr
        phase = 2.0 * math.pi * (day_of_year - 1.0) / 365.25
        return {
            "lr": torch.from_numpy(lr_scaled.copy()),
            "lr_raw": torch.from_numpy(
                np.nan_to_num(lr_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            ),
            "target_raw": torch.from_numpy(target_raw.astype(np.float32)),
            "static_lr": torch.from_numpy(static_lr),
            "static_hr": torch.from_numpy(static_hr),
            "valid_hr": torch.from_numpy(valid_hr[None].astype(np.float32)),
            "season": torch.tensor([math.sin(phase), math.cos(phase)], dtype=torch.float32),
            "time": torch.tensor([year, day_of_year], dtype=torch.int16),
        }
