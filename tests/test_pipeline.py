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
from climate_downscaling.stage2_data import Stage2FullFieldDataset, Stage2NetCDFPatchDataset
from climate_downscaling.stage2_prepare import prepare_stage2_index
from climate_downscaling.transforms import (
    TransformSpec,
    forward_numpy,
    inverse_numpy,
    specs_from_manifest,
)
from pipeline_02_train_mvswin import initialize_backbone, recover_early_stop_patience
from utility_plot_spatial_statistics import create_spatial_comparison_plots, plot_style


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


def create_synthetic_stage2(path: Path) -> Path:
    source = path / "source"
    dem = path / "dem"
    prepared = path / "prepared"
    lr_shape = (4, 5)
    hr_shape = (24, 30)
    transforms = {
        "tmin": TransformSpec("tmin", "standard", 0.0, 10.0),
        "tmax": TransformSpec("tmax", "standard", 5.0, 10.0),
        "prcp": TransformSpec("prcp", "log1p_standard", 0.5, 0.75),
    }
    stage1_manifest = path / "stage1_manifest.json"
    stage1_manifest.write_text(json.dumps({
        "transforms": {name: spec.to_dict() for name, spec in transforms.items()}
    }))
    for variable_index, variable in enumerate(VARIABLES):
        units = "K" if variable in {"tmin", "tmax"} else "mm/dy"
        for year in (2000, 2001, 2002):
            if variable == "tmin":
                lr = np.full((2, *lr_shape), 273.15, dtype=np.float32)
            elif variable == "tmax":
                lr = np.full((2, *lr_shape), 283.15, dtype=np.float32)
            else:
                lr = np.full((2, *lr_shape), 2.0, dtype=np.float32)
            hr = np.repeat(np.repeat(lr, 6, axis=-2), 6, axis=-1)
            if variable == "prcp":
                lr[0, 0, 0] = -2.0
                hr[0, 0, 0] = np.nan
            create_source_netcdf(
                source / f"Daymet_ERA5_{variable}_dy_{year}_0p25deg.nc",
                variable, lr, units,
            )
            create_source_netcdf(
                source / f"Daymet_ERA5_{variable}_dy_{year}_trim.nc",
                variable, hr, units,
            )
    create_dem_netcdf(
        dem / "VICa_DEM_0p25deg_fill0.nc", np.ones(lr_shape, dtype=np.float32)
    )
    create_dem_netcdf(
        dem / "VICa_DEM_trim_fill0.nc", np.ones(hr_shape, dtype=np.float32)
    )
    prepare_stage2_index(
        prepared,
        VARIABLES,
        {"train": [2000], "val": [2001], "test": [2002]},
        stage1_manifest,
        source,
        dem,
    )
    return prepared


class TransformTests(unittest.TestCase):
    def test_precipitation_round_trip(self) -> None:
        spec = TransformSpec("prcp", "log1p_standard", 0.5, 0.75)
        values = np.asarray([0.0, 1.0, 15.0], dtype=np.float32)
        restored = inverse_numpy(forward_numpy(values, spec), spec)
        np.testing.assert_allclose(restored, values, rtol=1e-5, atol=1e-5)

    def test_resume_recovers_early_stop_patience(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            records = [
                {"epoch": epoch, "validation": {"total": value}}
                for epoch, value in enumerate((0.5, 0.4, 0.41, 0.42, 0.43))
            ]
            history.write_text("\n".join(json.dumps(record) for record in records))
            checkpoint = {"epoch": 4, "best_validation": 0.4}
            self.assertEqual(recover_early_stop_patience(history, checkpoint), 3)
            checkpoint["early_stop_patience_count"] = 7
            self.assertEqual(recover_early_stop_patience(history, checkpoint), 7)


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

    def test_stage2_lazy_patch_and_full_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = create_synthetic_stage2(Path(directory))
            patch = Stage2NetCDFPatchDataset(
                prepared, "val", core_size=2, halo=1,
                patches_per_day=2, random_patches=False,
            )[0]
            self.assertEqual(tuple(patch["lr"].shape), (3, 4, 4))
            self.assertEqual(tuple(patch["target"].shape), (3, 24, 24))
            self.assertEqual(tuple(patch["static_hr"].shape), (4, 24, 24))

            full = Stage2FullFieldDataset(prepared, "test")
            sample = full[0]
            self.assertEqual(tuple(sample["lr"].shape), (3, 4, 5))
            self.assertEqual(tuple(sample["target_raw"].shape), (3, 24, 30))
            self.assertAlmostEqual(float(sample["lr_raw"][0, 0, 0]), 0.0, places=4)
            self.assertEqual(float(sample["lr_raw"][2, 0, 0]), 0.0)
            self.assertEqual(float(sample["valid_hr"][0, 0, 0]), 0.0)

            predictions_path = Path(directory) / "stage2_predictions.npy"
            predictions = np.lib.format.open_memmap(
                predictions_path, mode="w+", dtype=np.float32, shape=(2, 3, 24, 30)
            )
            for day in range(2):
                predictions[day] = full[day]["target_raw"].numpy()
            predictions.flush()
            output_dir = Path(directory) / "evaluation"
            paths = create_spatial_comparison_plots(
                prepared, predictions_path, output_dir, "test", VARIABLES, 2, tile_rows=7
            )
            self.assertEqual(len(paths), 7)
            statistics = np.load(output_dir / "spatial_statistics_1990.npz")
            self.assertAlmostEqual(
                float(np.nanmax(np.abs(statistics["mean_tmin_difference"]))), 0.0
            )

    def test_spatial_comparison_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_synthetic_dataset(root / "data")
            prediction_path = root / "predictions.npy"
            predictions = np.lib.format.open_memmap(
                prediction_path, mode="w+", dtype=np.float32, shape=(2, 3, 32, 40)
            )
            for channel, name in enumerate(VARIABLES):
                predictions[:, channel] = np.load(
                    root / "data" / "variables" / name / "hr_test.npy"
                )
            predictions.flush()
            paths = create_spatial_comparison_plots(
                root / "data", prediction_path, root / "evaluation", "test", VARIABLES, 2
            )
            self.assertEqual(len(paths), 7)
            self.assertTrue(all(path.exists() for path in paths))
            statistics = np.load(root / "evaluation" / "spatial_statistics_1990.npz")
            np.testing.assert_allclose(statistics["mean_tmin_difference"], 0.0)


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

    def test_requested_spatial_color_policy(self) -> None:
        self.assertEqual(plot_style("tmin", "mean", False), ("Spectral_r", (-20, 20)))
        self.assertEqual(plot_style("tmax", "p95", True), ("RdBu_r", (-5, 5)))
        self.assertEqual(plot_style("prcp", "p05", False), ("Spectral", (0, 1)))
        self.assertEqual(plot_style("prcp", "mean", True), ("RdBu", (-2, 2)))

    def test_sixfold_reconstruction_and_backbone_initialization(self) -> None:
        common = dict(
            embed_dim=24, num_groups=1, blocks_per_group=2,
            num_heads=4, window_size=4, variable_dropout=0.0, drop_path=0.0,
        )
        source = ClimateSwin(ClimateSwinConfig(scale_factor=4, **common))
        target = ClimateSwin(ClimateSwinConfig(scale_factor=6, **common))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.pt"
            torch.save({
                "model": source.state_dict(),
                "model_config": source.config.to_dict(),
            }, checkpoint)
            result = initialize_backbone(target, checkpoint)
        self.assertGreater(result["loaded_tensors"], 0)
        torch.testing.assert_close(
            target.variable_stem.stems[0].weight,
            source.variable_stem.stems[0].weight,
        )

        dynamic = torch.randn(1, 3, 7, 9)
        static_lr = torch.randn(1, 4, 7, 9)
        static_hr = torch.randn(1, 4, 42, 54)
        season = torch.randn(1, 2)
        prediction = target(dynamic, static_lr, static_hr, season)
        baseline = F.interpolate(dynamic, scale_factor=6, mode="bilinear", align_corners=False)
        self.assertEqual(tuple(prediction.shape), (1, 3, 42, 54))
        torch.testing.assert_close(prediction, baseline)

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
