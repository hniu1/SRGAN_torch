#!/usr/bin/env python3
"""Evaluate a ClimateSwin checkpoint on a chronological prepared-data split."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from climate_downscaling.data import FullFieldDataset
from climate_downscaling.model import ClimateSwin, ClimateSwinConfig, count_parameters
from climate_downscaling.transforms import inverse_channels_numpy
from utility_plot_spatial_statistics import create_spatial_comparison_plots


def load_evaluation_dataset(data_dir: Path, split: str, variable_names: tuple[str, ...]):
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    layout = manifest.get("storage_layout")
    if layout == "variable_separable_npy":
        return FullFieldDataset(data_dir, split=split, variable_names=variable_names), layout
    if layout == "netcdf_patch_index":
        from climate_downscaling.stage2_data import Stage2FullFieldDataset

        return Stage2FullFieldDataset(data_dir, split=split, variable_names=variable_names), layout
    raise ValueError(f"Unsupported evaluation storage layout: {layout!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data/daymet_mv_1980_1990"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/climateswin_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/climateswin_v1/test_1990"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--enforce-temperature-order", action="store_true")
    return parser


class ErrorAccumulator:
    def __init__(self, variable_names: tuple[str, ...]) -> None:
        self.variable_names = variable_names
        self.model_sum = np.zeros(len(variable_names), dtype=np.float64)
        self.model_abs = np.zeros(len(variable_names), dtype=np.float64)
        self.model_sq = np.zeros(len(variable_names), dtype=np.float64)
        self.baseline_sum = np.zeros(len(variable_names), dtype=np.float64)
        self.baseline_abs = np.zeros(len(variable_names), dtype=np.float64)
        self.baseline_sq = np.zeros(len(variable_names), dtype=np.float64)
        self.count = np.zeros(len(variable_names), dtype=np.int64)

    def update(
        self,
        prediction: np.ndarray,
        baseline: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        for channel in range(len(self.variable_names)):
            mask = valid & np.isfinite(prediction[channel]) & np.isfinite(target[channel])
            model_error = prediction[channel][mask].astype(np.float64) - target[channel][mask]
            baseline_error = baseline[channel][mask].astype(np.float64) - target[channel][mask]
            self.model_sum[channel] += model_error.sum()
            self.model_abs[channel] += np.abs(model_error).sum()
            self.model_sq[channel] += np.square(model_error).sum()
            self.baseline_sum[channel] += baseline_error.sum()
            self.baseline_abs[channel] += np.abs(baseline_error).sum()
            self.baseline_sq[channel] += np.square(baseline_error).sum()
            self.count[channel] += mask.sum()

    def summary(self) -> dict:
        result = {}
        for channel, name in enumerate(self.variable_names):
            count = max(int(self.count[channel]), 1)
            model = {
                "bias": self.model_sum[channel] / count,
                "mae": self.model_abs[channel] / count,
                "rmse": float(np.sqrt(self.model_sq[channel] / count)),
            }
            baseline = {
                "bias": self.baseline_sum[channel] / count,
                "mae": self.baseline_abs[channel] / count,
                "rmse": float(np.sqrt(self.baseline_sq[channel] / count)),
            }
            improvement = 100.0 * (baseline["mae"] - model["mae"]) / max(baseline["mae"], 1e-12)
            result[name] = {
                "model": model,
                "bilinear_baseline": baseline,
                "mae_improvement_percent": improvement,
                "count": count,
            }
        return result


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ClimateSwinConfig.from_dict(checkpoint["model_config"])
    dataset, storage_layout = load_evaluation_dataset(
        args.data_dir, args.split, config.variable_names
    )
    model = ClimateSwin(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_days = len(dataset) if args.max_days is None else min(len(dataset), args.max_days)
    predictions = None
    if not args.no_save_predictions:
        predictions = np.lib.format.open_memmap(
            args.output_dir / "predictions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_days, len(config.variable_names), *dataset.hr_shape),
        )

    default_valid = np.asarray(dataset.static_hr[-1] > 0.5)
    accumulator = ErrorAccumulator(config.variable_names)
    order_violations = 0
    valid_temperature_cells = 0
    output_index = 0
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            if output_index >= total_days:
                break
            batch_count = min(batch["lr"].shape[0], total_days - output_index)
            dynamic = batch["lr"][:batch_count].to(device)
            static_lr = batch["static_lr"][:batch_count].to(device)
            static_hr = batch["static_hr"][:batch_count].to(device)
            season = batch["season"][:batch_count].to(device)
            with amp_context():
                normalized_prediction = model(dynamic, static_lr, static_hr, season)
            normalized_prediction = normalized_prediction.float().cpu().numpy()
            target_raw = batch["target_raw"][:batch_count].numpy()
            valid_batch = batch.get("valid_hr")
            lr_raw = batch["lr_raw"][:batch_count].to(device)
            baseline_raw = F.interpolate(
                lr_raw, scale_factor=config.scale_factor, mode="bilinear", align_corners=False
            ).cpu().numpy()

            for batch_index in range(batch_count):
                valid = (
                    valid_batch[batch_index, 0].numpy() > 0.5
                    if valid_batch is not None else default_valid
                )
                prediction_raw = inverse_channels_numpy(
                    normalized_prediction[batch_index], config.variable_names, dataset.specs
                )
                if "tmin" in config.variable_names and "tmax" in config.variable_names:
                    i_min = config.variable_names.index("tmin")
                    i_max = config.variable_names.index("tmax")
                    invalid = prediction_raw[i_min] > prediction_raw[i_max]
                    order_violations += int((invalid & valid).sum())
                    valid_temperature_cells += int(valid.sum())
                    if args.enforce_temperature_order and invalid.any():
                        midpoint = 0.5 * (prediction_raw[i_min] + prediction_raw[i_max])
                        prediction_raw[i_min][invalid] = midpoint[invalid]
                        prediction_raw[i_max][invalid] = midpoint[invalid]
                accumulator.update(
                    prediction_raw, baseline_raw[batch_index], target_raw[batch_index], valid
                )
                if predictions is not None:
                    predictions[output_index] = prediction_raw
                output_index += 1
            if predictions is not None:
                predictions.flush()
            print(f"Evaluated {output_index}/{total_days} days", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "storage_layout": storage_layout,
        "days": output_index,
        "variables": list(config.variable_names),
        "variable_metadata": dataset.manifest.get("variable_metadata", {}),
        "parameters": count_parameters(model),
        "metrics": accumulator.summary(),
        "temperature_order_violation_fraction": (
            order_violations / max(valid_temperature_cells, 1)
        ),
        "temperature_order_enforced": args.enforce_temperature_order,
    }
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if predictions is not None and output_index > 0 and storage_layout == "variable_separable_npy":
        plot_paths = create_spatial_comparison_plots(
            data_dir=args.data_dir,
            predictions_path=args.output_dir / "predictions.npy",
            output_dir=args.output_dir,
            split=args.split,
            variable_names=config.variable_names,
            days=output_index,
        )
        summary["spatial_comparison_plots"] = [str(path) for path in plot_paths]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
