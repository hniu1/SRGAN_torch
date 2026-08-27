from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from climate_downscaling.data import FullFieldDataset, MultivariablePatchDataset
from climate_downscaling.losses import MultivariableLoss
from climate_downscaling.model import ClimateSwin, ClimateSwinConfig
from climate_downscaling.prepare import prepare_dataset, read_variable_diagnostics
from climate_downscaling.transforms import (
    TransformSpec,
    forward_numpy,
    inverse_numpy,
    specs_from_manifest,
)


VARIABLES = ("tmin", "tmax", "prcp")


def create_source_netcdf(
    path: Path, variable: str, values: np.ndarray, units: str | None = None
) -> None:
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", values.shape[0])
        dataset.createDimension("y", values.shape[1])
        dataset.createDimension("x", values.shape[2])
        field = dataset.createVariable(f"{variable}_dy", "f4", ("time", "y", "x"))
        field.units = units or ("mm/dy" if variable == "prcp" else "C")
        field.long_name = variable
        field[:] = values


def create_dem_netcdf(path: Path, values: np.ndarray) -> None:
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("y", values.shape[0])
        dataset.createDimension("x", values.shape[1])
        dataset.createVariable("DEM", "f4", ("y", "x"))[:] = values


def create_synthetic_dataset(path: Path) -> dict:
    lr_shape = (8, 10)
    hr_shape = (32, 40)
    transforms = {
        "tmin": TransformSpec("tmin", "standard", 0.0, 10.0),
        "tmax": TransformSpec("tmax", "standard", 5.0, 10.0),
        "prcp": TransformSpec("prcp", "log1p_standard", 0.5, 0.75),
    }
    manifest = {
        "format_version": 2,
        "storage_layout": "variable_separable_npy",
        "variables": list(VARIABLES),
        "scale_factor": 4,
        "lr_shape": list(lr_shape),
        "hr_shape": list(hr_shape),
        "splits": {name: {"years": [2000], "samples": 2} for name in ("train", "val", "test")},
        "transforms": {name: spec.to_dict() for name, spec in transforms.items()},
        "static": {"elevation_mean": 100.0, "elevation_std": 50.0},
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    shared = path / "shared"
    shared.mkdir()
    rng = np.random.default_rng(7)
    for split in ("train", "val", "test"):
        lr = rng.normal(size=(2, 3, *lr_shape)).astype(np.float32)
        lr[:, 0] = lr[:, 0] * 5.0
        lr[:, 1] = lr[:, 0] + 8.0
        lr[:, 2] = np.maximum(lr[:, 2], 0.0)
        hr = np.repeat(np.repeat(lr, 4, axis=-2), 4, axis=-1)
        for channel, name in enumerate(VARIABLES):
            variable_dir = path / "variables" / name
            variable_dir.mkdir(parents=True, exist_ok=True)
            np.save(variable_dir / f"lr_{split}.npy", lr[:, channel])
            np.save(variable_dir / f"hr_{split}.npy", hr[:, channel])
        np.save(shared / f"time_{split}.npy", np.asarray([[2000, 1], [2000, 180]], dtype=np.int16))
    np.save(shared / "elevation_lr.npy", np.linspace(0, 300, np.prod(lr_shape), dtype=np.float32).reshape(lr_shape))
    np.save(shared / "elevation_hr.npy", np.linspace(0, 300, np.prod(hr_shape), dtype=np.float32).reshape(hr_shape))
    y_lr, x_lr = np.meshgrid(
        np.linspace(-1, 1, lr_shape[0], dtype=np.float32),
        np.linspace(-1, 1, lr_shape[1], dtype=np.float32), indexing="ij",
    )
    y_hr, x_hr = np.meshgrid(
        np.linspace(-1, 1, hr_shape[0], dtype=np.float32),
        np.linspace(-1, 1, hr_shape[1], dtype=np.float32), indexing="ij",
    )
    np.save(shared / "coordinates_lr.npy", np.stack([y_lr, x_lr]))
    np.save(shared / "coordinates_hr.npy", np.stack([y_hr, x_hr]))
    for name in VARIABLES:
        variable_dir = path / "variables" / name
        np.save(variable_dir / "valid_lr.npy", np.ones(lr_shape, dtype=np.float32))
        np.save(variable_dir / "valid_hr.npy", np.ones(hr_shape, dtype=np.float32))
        (variable_dir / "complete.json").write_text(json.dumps({"variable": name}))
    return manifest


class TransformTests(unittest.TestCase):
    def test_precipitation_round_trip(self) -> None:
        spec = TransformSpec("prcp", "log1p_standard", 0.5, 0.75)
        values = np.asarray([0.0, 1.0, 15.0], dtype=np.float32)
        restored = inverse_numpy(forward_numpy(values, spec), spec)
        np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)


class DatasetTests(unittest.TestCase):
    def test_context_and_core_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            create_synthetic_dataset(Path(directory))
            dataset = MultivariablePatchDataset(
                Path(directory), "val", core_size=4, halo=2,
                patches_per_day=4, random_patches=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["lr"].shape), (3, 8, 8))
            self.assertEqual(tuple(sample["target"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["static_lr"].shape), (4, 8, 8))
            self.assertEqual(tuple(sample["static_hr"].shape), (4, 32, 32))
            self.assertEqual(float(sample["loss_mask"].sum()), 16 * 16)

    def test_selected_variables_are_stacked_lazily_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            create_synthetic_dataset(Path(directory))
            dataset = FullFieldDataset(
                Path(directory), "test", variable_names=("prcp", "tmin")
            )
            self.assertEqual(dataset.variable_names, ("prcp", "tmin"))
            self.assertEqual(tuple(dataset[0]["lr"].shape), (2, 8, 10))
            self.assertTrue(all(isinstance(array, np.memmap) for array in dataset.lr))


class PreparationTests(unittest.TestCase):
    def test_units_precipitation_bounds_and_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kelvin_path = root / "temperature.nc"
            precipitation_path = root / "precipitation.nc"
            create_source_netcdf(
                kelvin_path, "tmin",
                np.asarray([[[273.15, 300.0]]], dtype=np.float32), units="K",
            )
            create_source_netcdf(
                precipitation_path, "prcp",
                np.asarray([[[-2.0, np.nan, 5.0]]], dtype=np.float32), units="mm/dy",
            )

            temperature = read_variable_diagnostics(kelvin_path, "tmin")
            precipitation = read_variable_diagnostics(precipitation_path, "prcp")
            np.testing.assert_allclose(
                temperature.values, np.asarray([[[0.0, 26.85]]]), atol=1e-4
            )
            self.assertEqual(temperature.canonical_units, "degC")
            self.assertEqual(temperature.unit_conversion, "kelvin_to_celsius")
            self.assertEqual(precipitation.negative_clamped_count, 1)
            self.assertEqual(precipitation.missing_count, 1)
            self.assertEqual(float(precipitation.values[0, 0, 0]), 0.0)
            self.assertTrue(np.isnan(precipitation.values[0, 0, 1]))

    def test_new_variable_is_appended_without_rewriting_existing_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            dem = root / "dem"
            prepared = root / "prepared"
            lr_shape = (3, 4)
            hr_shape = (12, 16)
            years = (2000, 2001, 2002)
            for variable_index, variable in enumerate(("tmin", "prcp")):
                for year in years:
                    lr = np.full((2, *lr_shape), variable_index + year / 1000, dtype=np.float32)
                    hr = np.repeat(np.repeat(lr, 4, axis=-2), 4, axis=-1)
                    create_source_netcdf(
                        source / f"Daymet_ERA5_{variable}_dy_{year}_1deg.nc", variable, lr
                    )
                    create_source_netcdf(
                        source / f"Daymet_ERA5_{variable}_dy_{year}_0p25deg.nc", variable, hr
                    )
            create_dem_netcdf(dem / "VICa_DEM_1deg_fill0.nc", np.ones(lr_shape, dtype=np.float32))
            create_dem_netcdf(dem / "VICa_DEM_0p25deg_fill0.nc", np.ones(hr_shape, dtype=np.float32))
            splits = {"train": [2000], "val": [2001], "test": [2002]}

            prepare_dataset(prepared, ["tmin"], splits, source, dem)
            existing_path = prepared / "variables" / "tmin" / "lr_train.npy"
            existing_bytes = existing_path.read_bytes()
            prepare_dataset(prepared, ["prcp"], splits, source, dem)

            self.assertEqual(existing_path.read_bytes(), existing_bytes)
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertEqual(manifest["variables"], ["tmin", "prcp"])
            dataset = FullFieldDataset(prepared, "test")
            self.assertEqual(tuple(dataset[0]["lr"].shape), (2, *lr_shape))


class ModelTests(unittest.TestCase):
    def test_arbitrary_shape_and_zero_residual(self) -> None:
        config = ClimateSwinConfig(
            embed_dim=24,
            num_groups=1,
            blocks_per_group=2,
            num_heads=4,
            window_size=8,
            variable_dropout=0.0,
            drop_path=0.0,
        )
        model = ClimateSwin(config)
        dynamic = torch.randn(2, 3, 13, 17)
        static_lr = torch.randn(2, 4, 13, 17)
        static_hr = torch.randn(2, 4, 52, 68)
        season = torch.randn(2, 2)
        prediction = model(dynamic, static_lr, static_hr, season)
        baseline = F.interpolate(dynamic, scale_factor=4, mode="bilinear", align_corners=False)
        self.assertEqual(tuple(prediction.shape), (2, 3, 52, 68))
        torch.testing.assert_close(prediction, baseline)
        prediction.mean().backward()

    def test_loss_is_finite_and_differentiable(self) -> None:
        manifest = {"transforms": {
            "tmin": TransformSpec("tmin", "standard", 0.0, 10.0).to_dict(),
            "tmax": TransformSpec("tmax", "standard", 5.0, 10.0).to_dict(),
            "prcp": TransformSpec("prcp", "log1p_standard", 0.5, 0.75).to_dict(),
        }}
        objective = MultivariableLoss(VARIABLES, specs_from_manifest(manifest), scale_factor=4)
        prediction = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        dynamic = torch.randn(2, 3, 8, 8)
        mask = torch.ones(2, 1, 32, 32)
        loss, components = objective(prediction, target, dynamic, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("temperature_order", components)
        self.assertIn("precipitation_conservation", components)
        loss.backward()
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
