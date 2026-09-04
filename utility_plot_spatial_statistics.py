#!/usr/bin/env python3
"""Plot 1990 ClimateSwin temporal statistics against prepared Daymet truth."""

from __future__ import annotations

import argparse
import functools
import json
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


STATISTICS = (
    ("mean", "tmin", None),
    ("mean", "tmax", None),
    ("mean", "prcp", None),
    ("p95", "prcp", 95.0),
    ("p95", "tmax", 95.0),
    ("p05", "tmax", 5.0),
    ("p05", "tmin", 5.0),
)


def _temporal_statistic(values: np.ndarray, percentile: float | None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if percentile is None:
            return np.nanmean(values, axis=0, dtype=np.float64).astype(np.float32)
        return np.nanpercentile(values, percentile, axis=0).astype(np.float32)


def _unit_label(metadata: dict, variable: str) -> str:
    units = str(metadata.get(variable, {}).get("units", ""))
    return f" ({units})" if units else ""


def plot_style(variable: str, statistic: str, bias: bool) -> tuple[str, tuple[float, float]]:
    """Return the fixed colormap and range used across both downscaling stages."""
    variable = variable.lower()
    statistic_name = {"p95": "95th", "p05": "5th"}.get(statistic, statistic)
    if statistic_name not in {"mean", "95th", "5th"}:
        raise ValueError(f"Unknown statistic {statistic!r}")
    temperature = variable in {"tmax", "tmin", "tasmax", "tasmin"}
    if not bias:
        cmap = "Spectral_r" if temperature else "Spectral"
        if variable in {"pr", "prcp"}:
            defaults = {"mean": (0, 5), "95th": (0, 25), "5th": (0, 1)}
        elif variable in {"tmax", "tasmax"}:
            defaults = {"mean": (-5, 30), "95th": (0, 40), "5th": (-20, 20)}
        else:
            defaults = {"mean": (-20, 20), "95th": (0, 50), "5th": (-30, 15)}
    else:
        cmap = "RdBu_r" if temperature else "RdBu"
        if variable in {"pr", "prcp"}:
            defaults = {"mean": (-2, 2), "95th": (-8, 8), "5th": (-0.1, 0.1)}
        elif variable in {"tmax", "tasmax"}:
            defaults = {"mean": (-2, 2), "95th": (-5, 5), "5th": (-1, 1)}
        else:
            defaults = {"mean": (-2, 2), "95th": (-5, 5), "5th": (-5, 5)}
    return cmap, defaults[statistic_name]


def _grid_coordinates(manifest: dict, split: str, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the native one-dimensional longitude/latitude coordinates."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required to recover geographic coordinates") from exc
    if "source" not in manifest:
        height, width = (int(value) for value in manifest["hr_shape"])
        return np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    year = int(manifest["splits"][split]["years"][0])
    source = manifest["source"]
    if manifest.get("storage_layout") == "netcdf_patch_index":
        from climate_downscaling.stage2_prepare import stage2_source_path

        path = stage2_source_path(
            Path(source["data_root"]), variable, year, str(source["hr_suffix"])
        )
    else:
        path = Path(source["data_root"]) / f"Daymet_ERA5_{variable}_dy_{year}_0p25deg.nc"
    with Dataset(path) as dataset:
        if "lon" not in dataset.variables or "lat" not in dataset.variables:
            height, width = (int(value) for value in manifest["hr_shape"])
            return np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
    return longitude, latitude


def _image_extent(longitude: np.ndarray, latitude: np.ndarray) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(longitude)))
    dy = float(np.median(np.diff(latitude)))
    return (
        float(longitude[0] - 0.5 * dx),
        float(longitude[-1] + 0.5 * dx),
        float(latitude[0] - 0.5 * dy),
        float(latitude[-1] + 0.5 * dy),
    )


@functools.lru_cache(maxsize=2)
def _boundary_features(filename: str) -> tuple[dict, ...]:
    path = Path(__file__).resolve().parent / "assets" / "boundaries" / filename
    if not path.exists():
        return ()
    return tuple(json.loads(path.read_text())["features"])


def _geometry_rings(geometry: dict):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", ())
    if geometry_type == "Polygon":
        yield from coordinates
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon
    elif geometry_type == "LineString":
        yield coordinates
    elif geometry_type == "MultiLineString":
        yield from coordinates


def _plot_ring(axis, ring, **style) -> None:
    coordinates = np.asarray(ring, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        return
    longitude = coordinates[:, 0]
    longitude = np.where(longitude < 0.0, longitude + 360.0, longitude)
    latitude = coordinates[:, 1]
    jumps = np.flatnonzero(np.abs(np.diff(longitude)) > 180.0) + 1
    x_min, x_max = sorted(axis.get_xlim())
    y_min, y_max = sorted(axis.get_ylim())
    for segment in np.split(np.arange(longitude.size), jumps):
        if segment.size <= 1:
            continue
        x = longitude[segment]
        y = latitude[segment]
        if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
            continue
        axis.plot(x, y, **style)


def _add_boundaries(axis) -> None:
    country_features = _boundary_features("ne_50m_admin_0_countries.geojson")
    for feature in country_features:
        for ring in _geometry_rings(feature["geometry"]):
            _plot_ring(axis, ring, color="#202020", linewidth=0.55, alpha=0.9, zorder=4)
    state_features = _boundary_features("ne_50m_admin_1_states_provinces.geojson")
    for feature in state_features:
        if feature.get("properties", {}).get("adm0_a3") not in {"USA", "CAN", "MEX"}:
            continue
        for ring in _geometry_rings(feature["geometry"]):
            _plot_ring(axis, ring, color="#303030", linewidth=0.25, alpha=0.5, zorder=4)


def _format_map_axis(axis, extent: tuple[float, float, float, float]) -> None:
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value if value <= 180 else value - 360:.0f}°")
    )
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}°"))
    axis.tick_params(labelsize=8)
    _add_boundaries(axis)


class TruthTileReader:
    """Read prepared NPY or raw Stage-2 NetCDF truth one row tile at a time."""

    def __init__(self, data_dir: Path, manifest: dict, split: str, days: int) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = manifest
        self.split = split
        self.days = days
        self.layout = manifest.get("storage_layout")
        self.handles: dict[tuple[str, int], object] = {}
        if self.layout == "netcdf_patch_index":
            self.time = np.asarray(
                np.load(self.data_dir / "shared" / f"time_{split}.npy", mmap_mode="r")[:days]
            )
        elif self.layout != "variable_separable_npy":
            raise ValueError(f"Unsupported truth layout {self.layout!r}")

    def _stage2_field(self, variable: str, year: int):
        key = (variable, year)
        if key not in self.handles:
            try:
                from netCDF4 import Dataset
            except ImportError as exc:
                raise RuntimeError("netCDF4 is required for Stage-2 spatial plots") from exc
            from climate_downscaling.stage2_prepare import stage2_source_path

            source = self.manifest["source"]
            path = stage2_source_path(
                Path(source["data_root"]), variable, year, str(source["hr_suffix"])
            )
            self.handles[key] = Dataset(path)
        return self.handles[key].variables[f"{variable}_dy"]

    def read(self, variable: str, row0: int, row1: int) -> tuple[np.ndarray, np.ndarray]:
        if self.layout == "variable_separable_npy":
            truth = np.load(
                self.data_dir / "variables" / variable / f"hr_{self.split}.npy",
                mmap_mode="r",
            )[:self.days, row0:row1]
            valid = np.load(
                self.data_dir / "variables" / variable / "valid_hr.npy", mmap_mode="r"
            )[row0:row1] > 0.5
            values = np.asarray(truth, dtype=np.float32)
            return values, np.broadcast_to(valid, values.shape)

        width = int(self.manifest["hr_shape"][1])
        values = np.empty((self.days, row1 - row0, width), dtype=np.float32)
        valid = np.empty_like(values, dtype=bool)
        for year in np.unique(self.time[:, 0]):
            positions = np.flatnonzero(self.time[:, 0] == year)
            day_indices = self.time[positions, 1].astype(np.int64) - 1
            field = self._stage2_field(variable, int(year))
            raw = np.asarray(
                np.ma.filled(field[day_indices, row0:row1, :], np.nan), dtype=np.float32
            )
            current_valid = np.isfinite(raw)
            conversion = self.manifest.get("variable_metadata", {}).get(variable, {}).get(
                "unit_conversion"
            )
            if conversion == "kelvin_to_celsius":
                raw[current_valid] -= np.float32(273.15)
            if variable.lower() in {"pr", "prcp", "precip", "precipitation"}:
                raw[current_valid & (raw < 0.0)] = 0.0
            values[positions] = raw
            valid[positions] = current_valid
        return values, valid

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def create_spatial_comparison_plots(
    data_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    split: str,
    variable_names: Sequence[str],
    days: int,
    tile_rows: int = 21,
) -> list[Path]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    metadata = manifest.get("variable_metadata", {})
    predictions = np.load(predictions_path, mmap_mode="r")
    variable_names = tuple(variable_names)
    if predictions.shape[0] != days or predictions.shape[1] != len(variable_names):
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match {days} days and "
            f"variables {variable_names}"
        )

    height, width = (int(value) for value in predictions.shape[-2:])
    if tuple(manifest["hr_shape"]) != (height, width):
        raise ValueError(
            f"Prediction grid {(height, width)} does not match manifest {manifest['hr_shape']}"
        )
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    longitude, latitude = _grid_coordinates(manifest, split, variable_names[0])
    if longitude.shape != (width,) or latitude.shape != (height,):
        raise ValueError(
            f"Coordinate shapes {longitude.shape}, {latitude.shape} do not match {(height, width)}"
        )
    extent = _image_extent(longitude, latitude)

    plot_dir = output_dir / "spatial_statistics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    plot_paths: list[Path] = []

    reader = TruthTileReader(data_dir, manifest, split, days)
    try:
        for variable in variable_names:
            definitions = [entry for entry in STATISTICS if entry[1] == variable]
            if not definitions:
                continue
            channel = variable_names.index(variable)
            outputs = {
                (statistic, source): np.empty((height, width), dtype=np.float32)
                for statistic, _, _ in definitions
                for source in ("model", "daymet")
            }
            for row0 in range(0, height, tile_rows):
                row1 = min(row0 + tile_rows, height)
                truth_tile, valid = reader.read(variable, row0, row1)
                model_tile = np.asarray(
                    predictions[:days, channel, row0:row1, :], dtype=np.float32
                )
                model_tile = np.where(valid, model_tile, np.nan)
                truth_tile = np.where(valid, truth_tile, np.nan)
                for statistic, _, percentile in definitions:
                    outputs[(statistic, "model")][row0:row1] = _temporal_statistic(
                        model_tile, percentile
                    )
                    outputs[(statistic, "daymet")][row0:row1] = _temporal_statistic(
                        truth_tile, percentile
                    )
                print(f"[{variable}] spatial rows {row1}/{height}", flush=True)

            for statistic, _, _ in definitions:
                key = f"{statistic}_{variable}"
                arrays[f"{key}_model"] = outputs[(statistic, "model")]
                arrays[f"{key}_daymet"] = outputs[(statistic, "daymet")]
                arrays[f"{key}_difference"] = (
                    outputs[(statistic, "model")] - outputs[(statistic, "daymet")]
                )
    finally:
        reader.close()

    for statistic, variable, percentile in STATISTICS:
        if variable not in variable_names:
            continue
        key = f"{statistic}_{variable}"
        model_stat = arrays[f"{key}_model"]
        truth_stat = arrays[f"{key}_daymet"]
        difference = arrays[f"{key}_difference"]
        field_cmap, (field_min, field_max) = plot_style(variable, statistic, bias=False)
        bias_cmap, (bias_min, bias_max) = plot_style(variable, statistic, bias=True)
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
        for axis, values, title in (
            (axes[0], model_stat, "ClimateSwin"),
            (axes[1], truth_stat, "Daymet"),
        ):
            image = axis.imshow(
                values, origin="lower", cmap=field_cmap,
                vmin=field_min, vmax=field_max, interpolation="nearest", extent=extent,
            )
            axis.set_title(title)
            figure.colorbar(
                image, ax=axis, orientation="horizontal", pad=0.08,
                fraction=0.055, aspect=28,
            )
        image = axes[2].imshow(
            difference, origin="lower", cmap=bias_cmap,
            vmin=bias_min, vmax=bias_max, interpolation="nearest", extent=extent,
        )
        axes[2].set_title("ClimateSwin − Daymet")
        figure.colorbar(
            image, ax=axes[2], orientation="horizontal", pad=0.08,
            fraction=0.055, aspect=28,
        )
        label = "mean" if percentile is None else f"{int(percentile)}th percentile"
        figure.suptitle(f"1990 {variable} {label}{_unit_label(metadata, variable)}", fontsize=15)
        for axis in axes:
            _format_map_axis(axis, extent)
        path = plot_dir / f"{key}_comparison.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        plot_paths.append(path)

    np.savez_compressed(output_dir / "spatial_statistics_1990.npz", **arrays)
    index = {
        "split": split,
        "days": days,
        "variables": list(variable_names),
        "statistics": [path.stem.removesuffix("_comparison") for path in plot_paths],
        "plots": [str(path) for path in plot_paths],
        "arrays": str(output_dir / "spatial_statistics_1990.npz"),
        "color_scale_policy": "fixed variable/statistic ranges; physical fields are non-bias, model-minus-Daymet is bias",
        "boundaries": "Natural Earth 1:50m countries plus USA/Canada/Mexico states and provinces",
        "colorbars": "horizontal below each panel",
    }
    (output_dir / "spatial_statistics_index.json").write_text(json.dumps(index, indent=2) + "\n")
    summary_path = output_dir / "evaluation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["spatial_comparison_plots"] = [str(path.resolve()) for path in plot_paths]
        summary["spatial_statistics"] = str(
            (output_dir / "spatial_statistics_1990.npz").resolve()
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return plot_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tile-rows", type=int, default=21)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = json.loads((args.evaluation_dir / "evaluation_summary.json").read_text())
    paths = create_spatial_comparison_plots(
        data_dir=args.data_dir,
        predictions_path=args.evaluation_dir / "predictions.npy",
        output_dir=args.evaluation_dir,
        split=args.split,
        variable_names=summary["variables"],
        days=int(summary["days"]),
        tile_rows=args.tile_rows,
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
