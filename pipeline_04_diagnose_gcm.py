#!/usr/bin/env python3
"""
Run the trained Stage 1 SRGAN on a 1-degree GCM input and compare the
0.25-degree prediction against Daymet/ERA5 0.25-degree data.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset

from srgan_torch import SRGAN_g_lr_patch


DEFAULT_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data")
DEFAULT_GCM = Path(
    "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/gcm/bias-corrected/"
    "tmax_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_198001-201912_1deg_NA_BC.nc"
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare Stage 1 SRGAN 0.25-degree output against Daymet/ERA5."
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--version", default="tmax_stage1_patch_test")
    p.add_argument("--var", default="tmax")
    p.add_argument("--year", type=int, default=1980)
    p.add_argument("--gcm-path", type=Path, default=DEFAULT_GCM)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--n-days", type=int, default=366)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-elevation", action="store_true")
    p.add_argument("--save-prediction", action="store_true")
    return p


def choose_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def read_nc_var(path, var_name, n_days):
    with Dataset(path) as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{var_name!r} not found in {path}")
        arr = np.asarray(ds.variables[var_name][:n_days], dtype=np.float32)
    arr[np.isnan(arr)] = 0.0
    return arr


def read_daymet_hr(data_dir, var, year, n_days):
    path = data_dir / f"Daymet_ERA5_{var}_dy_{year}_0p25deg.nc"
    key = f"{var}_dy"
    return read_nc_var(path, key, n_days), path


def scale_array(arr, scaler):
    shp = arr.shape
    scaled = scaler.transform(arr.reshape(-1, 1)).reshape(*shp, 1)
    return scaled.astype(np.float32, copy=False)


def inverse_scale_array(arr, scaler, chunk_size=5_000_000):
    flat = arr.reshape(-1, 1)
    out = np.empty_like(flat, dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        out[start:end] = scaler.inverse_transform(flat[start:end])
    return out.reshape(arr.shape).astype(np.float32, copy=False)


def load_elevation(path_output, n_days):
    elev_path = path_output / "elev_lr_scaled.npy"
    config_path = path_output / "run_config.json"
    if not elev_path.exists() and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        elev_path = Path(config["data_path_output"]) / "elev_lr_scaled.npy"
    elev = np.load(elev_path, mmap_mode="r")
    elev2d = np.asarray(elev[0, :, :, 0], dtype=np.float32)
    return np.broadcast_to(elev2d[None, :, :, None], (n_days, *elev2d.shape, 1))


def run_stage1_model(args, gcm_lr, scaler, device):
    checkpoint_dir = args.base_dir / "models" / args.version
    path_output = args.base_dir / "output" / args.version

    lr_scaled = scale_array(gcm_lr, scaler)
    if not args.no_elevation:
        elev = load_elevation(path_output, lr_scaled.shape[0])
        lr_scaled = np.concatenate([lr_scaled, elev], axis=3)

    in_channels = lr_scaled.shape[3]
    model = SRGAN_g_lr_patch(in_channels=in_channels).to(device)
    state = torch.load(checkpoint_dir / "g.pth", map_location=device)
    model.load_state_dict(strip_module_prefix(state), strict=True)
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, lr_scaled.shape[0], args.batch_size):
            end = min(start + args.batch_size, lr_scaled.shape[0])
            batch = (
                torch.from_numpy(lr_scaled[start:end])
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device)
            )
            pred = model(batch).cpu().numpy()[:, 0, :, :]
            outputs.append(pred)
            del batch, pred

    pred_scaled = np.concatenate(outputs, axis=0)
    pred = inverse_scale_array(pred_scaled, scaler)
    if args.var in ("pr", "prcp"):
        np.maximum(pred, 0.0, out=pred)
    return pred


def metric_maps(pred, truth):
    diff = pred - truth
    return {
        "mean_pred": np.mean(pred, axis=0),
        "mean_truth": np.mean(truth, axis=0),
        "mean_diff": np.mean(diff, axis=0),
        "mae": np.mean(np.abs(diff), axis=0),
        "rmse": np.sqrt(np.mean(diff * diff, axis=0)),
        "bias_by_day": np.mean(diff, axis=(1, 2)),
        "mae_by_day": np.mean(np.abs(diff), axis=(1, 2)),
        "rmse_by_day": np.sqrt(np.mean(diff * diff, axis=(1, 2))),
    }


def imshow_clean(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data[::-1, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_day_comparison(plot_dir, gcm_lr, pred, truth, day_idx, var):
    day_idx = min(day_idx, pred.shape[0] - 1)
    diff = pred[day_idx] - truth[day_idx]
    vmin = float(min(np.nanpercentile(pred[day_idx], 2), np.nanpercentile(truth[day_idx], 2)))
    vmax = float(max(np.nanpercentile(pred[day_idx], 98), np.nanpercentile(truth[day_idx], 98)))
    dmax = float(np.nanpercentile(np.abs(diff), 98))
    dmax = max(dmax, 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    im0 = imshow_clean(axes[0], gcm_lr[day_idx], "GCM input 1 deg", "Spectral_r")
    im1 = imshow_clean(axes[1], pred[day_idx], "SRGAN pred 0.25 deg", "Spectral_r", vmin, vmax)
    im2 = imshow_clean(axes[2], truth[day_idx], "Daymet/ERA5 0.25 deg", "Spectral_r", vmin, vmax)
    im3 = imshow_clean(axes[3], diff, "Pred - truth", "RdBu_r", -dmax, dmax)
    fig.colorbar(im0, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.colorbar(im2, ax=axes[1:3], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.colorbar(im3, ax=axes[3], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.suptitle(f"{var} day index {day_idx}")
    fig.savefig(plot_dir / f"day_{day_idx:03d}_gcm_pred_truth_diff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_summary_maps(plot_dir, maps, var):
    pred = maps["mean_pred"]
    truth = maps["mean_truth"]
    diff = maps["mean_diff"]
    mae = maps["mae"]
    rmse = maps["rmse"]

    vmin = float(min(np.nanpercentile(pred, 2), np.nanpercentile(truth, 2)))
    vmax = float(max(np.nanpercentile(pred, 98), np.nanpercentile(truth, 98)))
    dmax = float(np.nanpercentile(np.abs(diff), 98))
    dmax = max(dmax, 1e-6)
    errmax = float(max(np.nanpercentile(mae, 98), np.nanpercentile(rmse, 98)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    imshow_clean(axes[0, 0], pred, "Mean prediction", "Spectral_r", vmin, vmax)
    im = imshow_clean(axes[0, 1], truth, "Mean truth", "Spectral_r", vmin, vmax)
    fig.colorbar(im, ax=axes[0, :2], orientation="horizontal", fraction=0.05, pad=0.06)
    im = imshow_clean(axes[0, 2], diff, "Mean bias", "RdBu_r", -dmax, dmax)
    fig.colorbar(im, ax=axes[0, 2], orientation="horizontal", fraction=0.05, pad=0.06)
    im = imshow_clean(axes[1, 0], mae, "MAE", "magma", 0, errmax)
    imshow_clean(axes[1, 1], rmse, "RMSE", "magma", 0, errmax)
    fig.colorbar(im, ax=axes[1, :2], orientation="horizontal", fraction=0.05, pad=0.06)
    axes[1, 2].axis("off")
    fig.suptitle(f"{var} Stage 1 GCM -> 0.25 deg comparison")
    fig.savefig(plot_dir / "summary_maps_mean_bias_mae_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(plot_dir, maps):
    days = np.arange(1, len(maps["mae_by_day"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, maps["bias_by_day"], color="#1f77b4", label="domain mean bias")
    ax.plot(days, maps["mae_by_day"], color="#d62728", label="domain MAE")
    ax.plot(days, maps["rmse_by_day"], color="#2ca02c", label="domain RMSE")
    ax.set_xlabel("Day")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Daily Domain-Average Error")
    fig.tight_layout()
    fig.savefig(plot_dir / "daily_error_timeseries.png", dpi=220)
    plt.close(fig)


def plot_histogram(plot_dir, pred, truth):
    diff = pred - truth
    rng = np.random.default_rng(42)
    vals = diff.reshape(-1)
    if vals.size > 500_000:
        vals = vals[rng.choice(vals.size, size=500_000, replace=False)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals[np.isfinite(vals)], bins=100, color="#666666", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Prediction - truth")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "error_histogram.png", dpi=220)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    checkpoint_dir = args.base_dir / "models" / args.version
    path_output = args.base_dir / "output" / args.version
    plot_dir = path_output / "plots" / "gcm_stage1_compare"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    with open(checkpoint_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    gcm_lr = read_nc_var(args.gcm_path, args.var, args.n_days)
    truth, truth_path = read_daymet_hr(args.data_dir, args.var, args.year, args.n_days)
    n_days = min(gcm_lr.shape[0], truth.shape[0], args.n_days)
    gcm_lr = gcm_lr[:n_days]
    truth = truth[:n_days]

    pred = run_stage1_model(args, gcm_lr, scaler, device)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

    if args.save_prediction:
        np.save(plot_dir / f"{args.var}_gcm_stage1_pred_0p25.npy", pred)

    maps = metric_maps(pred, truth)
    plot_day_comparison(plot_dir, gcm_lr, pred, truth, day_idx=0, var=args.var)
    plot_day_comparison(plot_dir, gcm_lr, pred, truth, day_idx=n_days // 2, var=args.var)
    plot_day_comparison(plot_dir, gcm_lr, pred, truth, day_idx=n_days - 1, var=args.var)
    plot_summary_maps(plot_dir, maps, args.var)
    plot_time_series(plot_dir, maps)
    plot_histogram(plot_dir, pred, truth)

    summary = {
        "version": args.version,
        "variable": args.var,
        "year": args.year,
        "n_days": int(n_days),
        "gcm_path": str(args.gcm_path),
        "truth_path": str(truth_path),
        "prediction_shape": [int(v) for v in pred.shape],
        "truth_shape": [int(v) for v in truth.shape],
        "mean_bias": float(np.mean(pred - truth)),
        "mean_mae": float(np.mean(np.abs(pred - truth))),
        "mean_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
    }
    with open(plot_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote comparison plots to {plot_dir}", flush=True)
    for path in sorted(plot_dir.glob("*")):
        print(path.name, flush=True)


if __name__ == "__main__":
    main()
