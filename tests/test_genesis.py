"""Verify both 5x stages, TVA coordinates, masks, dates and checkpoint isolation."""
from __future__ import annotations

import datetime as dt
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from netCDF4 import Dataset, date2num
import numpy as np
import torch
import torch.nn.functional as F

import pipeline_04_infer
import pipeline_02_train
import pipeline_03_evaluate
from refine_downscaling.data import validate_checkpoint_manifest
from refine_downscaling.daymet_prepare import daymet_source_path, prepare_daymet_index, read_dates
from refine_downscaling.losses import MultivariableLoss
from refine_downscaling.model import REFINE, REFINEConfig
from refine_downscaling.stage2_data import Stage2FullFieldDataset, Stage2NetCDFPatchDataset
from refine_downscaling.transforms import specs_from_manifest
from utility_plot_spatial_statistics import TruthTileReader


VARIABLES = ("tmin", "tmax", "prcp")
SPLITS = {"train": [2000], "val": [2001], "test": [2002]}


def create_tva_source(root: Path) -> dict[str, Path]:
    terrain = {}
    for res, factor, spacing in (("025deg", 1, 0.25), ("005deg", 5, 0.05), ("001deg", 25, 0.01)):
        height, width = 2 * factor, 3 * factor
        lat = 32.5 + (np.arange(height) + 0.5) * spacing
        lon = -91 + (np.arange(width) + 0.5) * spacing
        dem_path = root / f"dem_{res}.nc"
        with Dataset(dem_path, "w") as ds:
            ds.createDimension("lat", height)
            ds.createDimension("lon", width)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            elevation = np.arange(height * width, dtype=np.float32).reshape(height, width)
            elevation[1, 1] = np.nan
            ds.createVariable("elevation", "f4", ("lat", "lon"))[:] = elevation
        terrain[res] = dem_path
        for variable in VARIABLES:
            for year in (2000, 2001, 2002):
                path = daymet_source_path(root / "source", variable, year, res)
                path.parent.mkdir(parents=True, exist_ok=True)
                values = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None, None] + 1, (3, height, width)).copy()
                if variable == "tmax":
                    values += 10
                if year == 2002:
                    values += 100  # Test values must not influence fitted normalization.
                values[:, 0, 0] = np.nan
                with Dataset(path, "w") as ds:
                    ds.createDimension("time", 3)
                    ds.createDimension("lat", height)
                    ds.createDimension("lon", width)
                    ds.createVariable("lat", "f8", ("lat",))[:] = lat
                    ds.createVariable("lon", "f8", ("lon",))[:] = lon
                    time = ds.createVariable("time", "f8", ("time",))
                    time.units, time.calendar = f"days since {year}-01-01", "standard"
                    dates = [dt.datetime(year, 1, 1, 12) + dt.timedelta(days=d) for d in (0, 59, 61)]
                    time[:] = date2num(dates, time.units, time.calendar)
                    field = ds.createVariable(variable, "f4", ("time", "lat", "lon"), fill_value=np.nan)
                    field.units = "mm/day" if variable == "prcp" else "degrees C"
                    field.long_name = variable
                    field[:] = values
    return terrain


class GenesisTests(unittest.TestCase):
    def prepare(self, root: Path):
        terrain = create_tva_source(root)
        paths = []
        for stage, lr, hr in ((1, "025deg", "005deg"), (2, "005deg", "001deg")):
            prepared = root / f"stage{stage}"
            prepare_daymet_index(prepared, VARIABLES, SPLITS, root / "source", stage,
                                 terrain[lr], terrain[hr])
            paths.append(prepared)
        return paths

    def test_both_stages_training_statistics_dates_masks_and_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            for stage, prepared in enumerate(self.prepare(Path(temporary)), 1):
                dataset = Stage2FullFieldDataset(prepared, "train")
                self.assertEqual(dataset.scale_factor, 5)
                self.assertEqual(dataset.manifest["stage"], stage)
                self.assertAlmostEqual(dataset.specs["tmin"].mean, 2)
                self.assertAlmostEqual(dataset.specs["tmax"].mean, 12)
                self.assertEqual(dataset[2]["time"].tolist(), [2000, 62])
                self.assertEqual(float(dataset[2]["lr_raw"][0, 0, 1]), 3)
                self.assertEqual(float(dataset[0]["valid_hr"][0, 0, 0]), 0)
                self.assertEqual(float(dataset[0]["valid_hr"][0, 1, 1]), 0)
                reader = TruthTileReader(prepared, dataset.manifest, "train", 3)
                values, valid = reader.read("tmin", 0, 2)
                self.assertEqual(float(values[2, 0, 1]), 3)
                self.assertFalse(valid[0, 0, 0])
                reader.close()
                patches = Stage2NetCDFPatchDataset(prepared, "train", core_size=2, halo=1,
                                                   patches_per_day=1, random_patches=False)
                sample = patches[2]
                self.assertEqual(tuple(sample["lr"].shape), (3, 4, 4))
                self.assertEqual(tuple(sample["target"].shape), (3, 20, 20))
                self.assertTrue(torch.isfinite(sample["target"]).all())
                model = REFINE(REFINEConfig(scale_factor=5, embed_dim=12, num_heads=3,
                                           num_groups=1, blocks_per_group=1, window_size=4,
                                           variable_dropout=0, drop_path=0))
                dynamic = sample["lr"][None]
                prediction = model(dynamic, sample["static_lr"][None], sample["static_hr"][None], sample["season"][None])
                torch.testing.assert_close(prediction, F.interpolate(dynamic, scale_factor=5, mode="bilinear", align_corners=False))
                objective = MultivariableLoss(VARIABLES, specs_from_manifest(dataset.manifest), scale_factor=5)
                loss, _ = objective(prediction, sample["target"][None], dynamic, sample["loss_mask"][None])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.upsampling[0].expand.weight.grad)
                patches.close()
                dataset.close()

    def test_one_epoch_training_and_evaluation_for_each_stage(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            for stage, prepared in enumerate(self.prepare(root), 1):
                run = root / f"run{stage}"
                argv = ["train", "--data-dir", str(prepared), "--run-dir", str(run),
                        "--epochs", "1", "--batch-size", "1", "--patches-per-day", "1",
                        "--validation-patches-per-day", "1", "--core-size", "2", "--halo", "1",
                        "--embed-dim", "12", "--num-heads", "3", "--num-groups", "1",
                        "--blocks-per-group", "1", "--window-size", "4"]
                with patch("sys.argv", argv), patch("torch.cuda.is_available", return_value=False):
                    pipeline_02_train.main()
                checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["model_config"]["scale_factor"], 5)
                self.assertEqual(checkpoint["data_manifest"]["stage"], stage)
                self.assertTrue(torch.isfinite(checkpoint["model"]["decoders.0.output.weight"]).all())
                self.assertTrue(checkpoint["model"]["decoders.0.output.weight"].abs().sum() > 0)
                evaluation = root / f"evaluation{stage}"
                argv = ["evaluate", "--data-dir", str(prepared), "--checkpoint", str(run / "best.pt"),
                        "--output-dir", str(evaluation), "--max-days", "1", "--no-save-predictions"]
                with patch("sys.argv", argv), patch("torch.cuda.is_available", return_value=False):
                    pipeline_03_evaluate.main()
                summary = json.loads((evaluation / "evaluation_summary.json").read_text())
                self.assertEqual(summary["days"], 1)
                self.assertGreater(summary["metrics"]["tmin"]["count"], 0)

    def test_chained_inference_preserves_coordinates_timestamps_and_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.prepare(root)
            upstream = None
            for stage, prepared in enumerate(paths, 1):
                manifest = json.loads((prepared / "manifest.json").read_text())
                model = REFINE(REFINEConfig(scale_factor=5, embed_dim=12, num_heads=3,
                                           num_groups=1, blocks_per_group=1, window_size=4))
                checkpoint = root / f"model{stage}.pt"
                torch.save({"model_config": model.config.to_dict(), "model": model.state_dict(),
                            "data_manifest": manifest}, checkpoint)
                if stage == 1:
                    with self.assertRaisesRegex(ValueError, "differ"):
                        validate_checkpoint_manifest(torch.load(checkpoint, weights_only=False),
                                                     json.loads((paths[1] / "manifest.json").read_text()))
                output = root / f"result{stage}.nc"
                argv = ["infer", "--data-dir", str(prepared), "--checkpoint", str(checkpoint),
                        "--output", str(output), "--end-index", "3", "--enforce-temperature-order"]
                for name in VARIABLES:
                    source = upstream or daymet_source_path(root / "source", name, 2002, "025deg")
                    argv += ["--input", f"{name}={source}"]
                with patch("sys.argv", argv), patch("torch.cuda.is_available", return_value=False):
                    pipeline_04_infer.main()
                with Dataset(output) as ds:
                    self.assertEqual(ds.variables["tmin"].dimensions, ("time", "lat", "lon"))
                    self.assertEqual(list(ds.variables["tmin"].shape[-2:]), manifest["hr_shape"])
                    self.assertTrue(np.ma.getmaskarray(ds.variables["tmin"][:])[:, 0, 0].all())
                    self.assertEqual(read_dates(ds)[2], dt.datetime(2002, 3, 3, 12))
                    for axis in ("lat", "lon"):
                        np.testing.assert_array_equal(ds.variables[axis][:], np.load(prepared / "shared" / f"{axis}_hr.npy"))
                upstream = output
            with Dataset(upstream) as ds:
                self.assertEqual(ds.variables["tmin"].shape, (3, 50, 75))

    def test_rejects_shifted_grid_and_overlapping_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terrain = create_tva_source(root)
            shifted = daymet_source_path(root / "source", "tmax", 2001, "005deg")
            with Dataset(shifted, "a") as ds:
                ds.variables["lon"][:] += 0.01
            with self.assertRaisesRegex(ValueError, "coordinates"):
                prepare_daymet_index(root / "prepared", VARIABLES, SPLITS, root / "source", 1,
                                     terrain["025deg"], terrain["005deg"])
            with self.assertRaisesRegex(ValueError, "chronologically"):
                prepare_daymet_index(root / "prepared", VARIABLES,
                                     {"train": [2000], "val": [2000], "test": [2002]},
                                     root / "source", 1, terrain["025deg"], terrain["005deg"])


if __name__ == "__main__":
    unittest.main()
