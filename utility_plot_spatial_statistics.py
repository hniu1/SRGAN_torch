#!/usr/bin/env python3
"""Plot 1990 ClimateSwin temporal statistics against prepared Daymet truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    if percentile is None:
        return np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    return np.percentile(values, percentile, axis=0).astype(np.float32)


def _unit_label(metadata: dict, variable: str) -> str:
    units = str(metadata.get(variable, {}).get("units", ""))
    return f" ({units})" if units else ""


def _field_limits(model: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    lower = min(float(np.nanmin(model)), float(np.nanmin(truth)))
    upper = max(float(np.nanmax(model)), float(np.nanmax(truth)))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def create_spatial_comparison_plots(
    data_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    split: str,
    variable_names: Sequence[str],
    days: int,
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

    plot_dir = output_dir / "spatial_statistics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    plot_paths: list[Path] = []

    for statistic, variable, percentile in STATISTICS:
        if variable not in variable_names:
            continue
        channel = variable_names.index(variable)
        truth_path = data_dir / "variables" / variable / f"hr_{split}.npy"
        truth = np.load(truth_path, mmap_mode="r")
        if truth.shape[0] < days or tuple(truth.shape[-2:]) != tuple(predictions.shape[-2:]):
            raise ValueError(f"Daymet shape {truth.shape} is incompatible with predictions")
        valid = np.load(data_dir / "variables" / variable / "valid_hr.npy") > 0.5

        model_stat = _temporal_statistic(predictions[:days, channel], percentile)
        truth_stat = _temporal_statistic(truth[:days], percentile)
        model_stat[~valid] = np.nan
        truth_stat[~valid] = np.nan
        difference = model_stat - truth_stat
        key = f"{statistic}_{variable}"
        arrays[f"{key}_model"] = model_stat
        arrays[f"{key}_daymet"] = truth_stat
        arrays[f"{key}_difference"] = difference

        field_min, field_max = _field_limits(model_stat, truth_stat)
        difference_limit = float(np.nanmax(np.abs(difference)))
        difference_limit = max(difference_limit, 1e-6)
        field_cmap = "YlGnBu" if variable == "prcp" else "coolwarm"
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        for axis, values, title in (
            (axes[0], model_stat, "ClimateSwin"),
            (axes[1], truth_stat, "Daymet"),
        ):
            image = axis.imshow(
                values, origin="lower", cmap=field_cmap,
                vmin=field_min, vmax=field_max, interpolation="nearest",
            )
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.82)
        image = axes[2].imshow(
            difference, origin="lower", cmap="RdBu_r",
            vmin=-difference_limit, vmax=difference_limit, interpolation="nearest",
        )
        axes[2].set_title("ClimateSwin − Daymet")
        figure.colorbar(image, ax=axes[2], shrink=0.82)
        label = "mean" if percentile is None else f"{int(percentile)}th percentile"
        figure.suptitle(f"1990 {variable} {label}{_unit_label(metadata, variable)}", fontsize=15)
        for axis in axes:
            axis.set_xlabel("Grid x")
            axis.set_ylabel("Grid y")
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
    }
    (output_dir / "spatial_statistics_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return plot_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
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
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
