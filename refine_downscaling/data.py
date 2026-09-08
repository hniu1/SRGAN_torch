"""Memory-mapped patch and full-field datasets for the joint pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import TransformSpec, specs_from_manifest, transform_channels_numpy


def load_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared-data manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 2 or manifest.get("storage_layout") != "variable_separable_npy":
        raise ValueError(f"Unsupported prepared-data layout in {path}")
    return manifest


def _select_variables(manifest: dict, requested: Sequence[str] | None) -> tuple[str, ...]:
    available = tuple(str(name) for name in manifest["variables"])
    selected = available if requested is None else tuple(str(name) for name in requested)
    if not selected:
        raise ValueError("At least one prepared variable must be selected")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Variables are not prepared: {sorted(missing)}; available={list(available)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected variables cannot contain duplicates")
    return selected


def _require_complete_variables(data_dir: Path, variable_names: Sequence[str]) -> None:
    incomplete = [
        name for name in variable_names
        if not (data_dir / "variables" / name / "complete.json").exists()
    ]
    if incomplete:
        raise RuntimeError(f"Prepared variables are incomplete: {incomplete}")


def _crop_with_reflection(
    array: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Crop the final two axes and reflect-pad portions outside the domain."""
    source_h, source_w = array.shape[-2:]
    top = max(-row, 0)
    left = max(-col, 0)
    bottom = max(row + height - source_h, 0)
    right = max(col + width - source_w, 0)
    row0, row1 = max(row, 0), min(row + height, source_h)
    col0, col1 = max(col, 0), min(col + width, source_w)
    cropped = np.asarray(array[..., row0:row1, col0:col1])
    if top or bottom or left or right:
        pad = [(0, 0)] * cropped.ndim
        pad[-2] = (top, bottom)
        pad[-1] = (left, right)
        cropped = np.pad(cropped, pad, mode="reflect")
    if cropped.shape[-2:] != (height, width):
        raise RuntimeError(f"Crop produced {cropped.shape[-2:]}, expected {(height, width)}")
    return cropped


def _validation_origins(height: int, width: int, core_size: int, count: int) -> list[Tuple[int, int]]:
    n_rows = max(1, int(round(math.sqrt(count * height / max(width, 1)))))
    n_cols = max(1, int(math.ceil(count / n_rows)))
    rows = np.linspace(0, max(height - core_size, 0), n_rows).round().astype(int)
    cols = np.linspace(0, max(width - core_size, 0), n_cols).round().astype(int)
    origins = [(int(row), int(col)) for row in rows for col in cols]
    if len(origins) == 1:
        return origins * count
    selected = np.linspace(0, len(origins) - 1, count).round().astype(int)
    return [origins[index] for index in selected]


class MultivariablePatchDataset(Dataset):
    """Sample context patches while supervising only a central, seam-safe core."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        core_size: int = 16,
        halo: int = 4,
        patches_per_day: int = 16,
        random_patches: bool = True,
        variable_names: Sequence[str] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if core_size <= 0 or halo < 0 or patches_per_day <= 0:
            raise ValueError("core_size and patches_per_day must be positive; halo cannot be negative")
        self.data_dir = Path(data_dir)
        self.manifest = load_manifest(self.data_dir)
        self.variable_names = _select_variables(self.manifest, variable_names)
        _require_complete_variables(self.data_dir, self.variable_names)
        self.specs: Dict[str, TransformSpec] = specs_from_manifest(self.manifest)
        self.scale_factor = int(self.manifest["scale_factor"])
        self.split = split
        self.core_size = int(core_size)
        self.halo = int(halo)
        self.context_size = self.core_size + 2 * self.halo
        self.patches_per_day = int(patches_per_day)
        self.random_patches = bool(random_patches)

        self.lr = tuple(
            np.load(self.data_dir / "variables" / name / f"lr_{split}.npy", mmap_mode="r")
            for name in self.variable_names
        )
        self.hr = tuple(
            np.load(self.data_dir / "variables" / name / f"hr_{split}.npy", mmap_mode="r")
            for name in self.variable_names
        )
        shared_dir = self.data_dir / "shared"
        self.time = np.load(shared_dir / f"time_{split}.npy", mmap_mode="r")
        self.elevation_lr = np.load(shared_dir / "elevation_lr.npy", mmap_mode="r")
        self.elevation_hr = np.load(shared_dir / "elevation_hr.npy", mmap_mode="r")
        self.coordinates_lr = np.load(shared_dir / "coordinates_lr.npy", mmap_mode="r")
        self.coordinates_hr = np.load(shared_dir / "coordinates_hr.npy", mmap_mode="r")
        valid_lr_fields = [
            np.load(self.data_dir / "variables" / name / "valid_lr.npy", mmap_mode="r")
            for name in self.variable_names
        ]
        valid_hr_fields = [
            np.load(self.data_dir / "variables" / name / "valid_hr.npy", mmap_mode="r")
            for name in self.variable_names
        ]
        self.valid_lr = np.logical_and.reduce([field > 0.5 for field in valid_lr_fields]).astype(np.float32)
        self.valid_hr = np.logical_and.reduce([field > 0.5 for field in valid_hr_fields]).astype(np.float32)

        sample_counts = {array.shape[0] for array in (*self.lr, *self.hr)}
        if sample_counts != {self.time.shape[0]}:
            raise ValueError(f"Prepared variable/time dimensions are inconsistent: {sample_counts}, {self.time.shape[0]}")
        self.lr_shape = tuple(int(v) for v in self.lr[0].shape[-2:])
        self.hr_shape = tuple(int(v) for v in self.hr[0].shape[-2:])
        if any(array.shape[-2:] != self.lr_shape for array in self.lr):
            raise ValueError("Prepared LR variable grids do not match")
        if any(array.shape[-2:] != self.hr_shape for array in self.hr):
            raise ValueError("Prepared HR variable grids do not match")
        if self.hr_shape != tuple(v * self.scale_factor for v in self.lr_shape):
            raise ValueError(f"Prepared spatial dimensions are inconsistent: {self.lr_shape}, {self.hr_shape}")
        if core_size > min(self.lr_shape):
            raise ValueError(f"core_size={core_size} exceeds LR field {self.lr_shape}")

        static = self.manifest["static"]
        self.elevation_mean = float(static["elevation_mean"])
        self.elevation_std = float(static["elevation_std"])
        self._deterministic_origins = _validation_origins(
            self.lr_shape[0], self.lr_shape[1], self.core_size, self.patches_per_day
        )

    def __len__(self) -> int:
        return int(self.lr[0].shape[0]) * self.patches_per_day

    def _origin(self, patch_index: int) -> Tuple[int, int]:
        if not self.random_patches:
            return self._deterministic_origins[patch_index]
        row = int(np.random.randint(0, self.lr_shape[0] - self.core_size + 1))
        col = int(np.random.randint(0, self.lr_shape[1] - self.core_size + 1))
        return row, col

    def _static_patch(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        lr_elev = _crop_with_reflection(
            self.elevation_lr, row, col, self.context_size, self.context_size
        )[None]
        lr_coord = _crop_with_reflection(
            self.coordinates_lr, row, col, self.context_size, self.context_size
        )
        lr_valid = _crop_with_reflection(
            self.valid_lr, row, col, self.context_size, self.context_size
        )[None]
        lr_elev = (lr_elev.astype(np.float32) - self.elevation_mean) / self.elevation_std
        static_lr = np.concatenate([lr_elev, lr_coord.astype(np.float32), lr_valid.astype(np.float32)])

        factor = self.scale_factor
        hr_size = self.context_size * factor
        hr_row, hr_col = row * factor, col * factor
        hr_elev = _crop_with_reflection(self.elevation_hr, hr_row, hr_col, hr_size, hr_size)[None]
        hr_coord = _crop_with_reflection(self.coordinates_hr, hr_row, hr_col, hr_size, hr_size)
        hr_valid = _crop_with_reflection(self.valid_hr, hr_row, hr_col, hr_size, hr_size)[None]
        hr_elev = (hr_elev.astype(np.float32) - self.elevation_mean) / self.elevation_std
        static_hr = np.concatenate([hr_elev, hr_coord.astype(np.float32), hr_valid.astype(np.float32)])
        return static_lr, static_hr

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        day = index // self.patches_per_day
        patch_index = index % self.patches_per_day
        core_row, core_col = self._origin(patch_index)
        context_row = core_row - self.halo
        context_col = core_col - self.halo

        lr_raw = np.stack([
            _crop_with_reflection(
                array[day], context_row, context_col, self.context_size, self.context_size
            )
            for array in self.lr
        ]).astype(np.float32)
        factor = self.scale_factor
        hr_size = self.context_size * factor
        hr_raw = np.stack([
            _crop_with_reflection(
                array[day], context_row * factor, context_col * factor, hr_size, hr_size
            )
            for array in self.hr
        ]).astype(np.float32)
        lr_scaled = transform_channels_numpy(lr_raw, self.variable_names, self.specs)
        hr_scaled = transform_channels_numpy(hr_raw, self.variable_names, self.specs)
        static_lr, static_hr = self._static_patch(context_row, context_col)

        loss_mask = np.zeros((1, hr_size, hr_size), dtype=np.float32)
        margin = self.halo * factor
        core_hr = self.core_size * factor
        loss_mask[:, margin:margin + core_hr, margin:margin + core_hr] = 1.0
        loss_mask *= static_hr[-1:]

        day_of_year = float(self.time[day, 1])
        phase = 2.0 * math.pi * (day_of_year - 1.0) / 365.25
        season = np.asarray([math.sin(phase), math.cos(phase)], dtype=np.float32)
        return {
            "lr": torch.from_numpy(lr_scaled.copy()),
            "target": torch.from_numpy(hr_scaled.copy()),
            "static_lr": torch.from_numpy(static_lr.copy()),
            "static_hr": torch.from_numpy(static_hr.copy()),
            "loss_mask": torch.from_numpy(loss_mask),
            "season": torch.from_numpy(season),
        }


class FullFieldDataset(Dataset):
    """Return complete days for evaluation or memory-safe batched inference."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "test",
        variable_names: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = load_manifest(self.data_dir)
        self.variable_names = _select_variables(self.manifest, variable_names)
        _require_complete_variables(self.data_dir, self.variable_names)
        self.specs = specs_from_manifest(self.manifest)
        self.split = split
        self.lr = tuple(
            np.load(self.data_dir / "variables" / name / f"lr_{split}.npy", mmap_mode="r")
            for name in self.variable_names
        )
        self.hr = tuple(
            np.load(self.data_dir / "variables" / name / f"hr_{split}.npy", mmap_mode="r")
            for name in self.variable_names
        )
        shared_dir = self.data_dir / "shared"
        self.time = np.load(shared_dir / f"time_{split}.npy", mmap_mode="r")
        self.scale_factor = int(self.manifest["scale_factor"])
        self.lr_shape = tuple(int(value) for value in self.manifest["lr_shape"])
        self.hr_shape = tuple(int(value) for value in self.manifest["hr_shape"])
        counts = {array.shape[0] for array in (*self.lr, *self.hr)}
        if counts != {self.time.shape[0]}:
            raise ValueError(f"Prepared variable/time dimensions are inconsistent: {counts}, {self.time.shape[0]}")

        elevation_mean = float(self.manifest["static"]["elevation_mean"])
        elevation_std = float(self.manifest["static"]["elevation_std"])
        elev_lr = np.load(shared_dir / "elevation_lr.npy").astype(np.float32)
        elev_hr = np.load(shared_dir / "elevation_hr.npy").astype(np.float32)
        coords_lr = np.load(shared_dir / "coordinates_lr.npy").astype(np.float32)
        coords_hr = np.load(shared_dir / "coordinates_hr.npy").astype(np.float32)
        valid_lr = np.logical_and.reduce([
            np.load(self.data_dir / "variables" / name / "valid_lr.npy") > 0.5
            for name in self.variable_names
        ]).astype(np.float32)
        valid_hr = np.logical_and.reduce([
            np.load(self.data_dir / "variables" / name / "valid_hr.npy") > 0.5
            for name in self.variable_names
        ]).astype(np.float32)
        self.static_lr = np.concatenate([
            ((elev_lr - elevation_mean) / elevation_std)[None], coords_lr, valid_lr[None]
        ]).astype(np.float32)
        self.static_hr = np.concatenate([
            ((elev_hr - elevation_mean) / elevation_std)[None], coords_hr, valid_hr[None]
        ]).astype(np.float32)

    def __len__(self) -> int:
        return int(self.lr[0].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        lr_raw = np.stack([np.asarray(array[index], dtype=np.float32) for array in self.lr])
        target_raw = np.stack([np.asarray(array[index], dtype=np.float32) for array in self.hr])
        day_of_year = float(self.time[index, 1])
        phase = 2.0 * math.pi * (day_of_year - 1.0) / 365.25
        return {
            "lr": torch.from_numpy(transform_channels_numpy(lr_raw, self.variable_names, self.specs).copy()),
            "lr_raw": torch.from_numpy(lr_raw.copy()),
            "target_raw": torch.from_numpy(target_raw.copy()),
            "static_lr": torch.from_numpy(self.static_lr.copy()),
            "static_hr": torch.from_numpy(self.static_hr.copy()),
            "valid_hr": torch.from_numpy(self.static_hr[-1:].copy()),
            "season": torch.tensor([math.sin(phase), math.cos(phase)], dtype=torch.float32),
            "time": torch.from_numpy(np.asarray(self.time[index], dtype=np.int16).copy()),
        }
