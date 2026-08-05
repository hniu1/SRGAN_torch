#!/usr/bin/env python3
"""Evaluate Stage-1 downscaling on an entirely unseen Daymet/ERA5 year."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset

from srgan_torch import SRGAN_g_lr_patch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data")


def parser():
    p = argparse.ArgumentParser(description="Paired Daymet 1deg -> 0.25deg evaluation")
    p.add_argument("--base-dir", type=Path, default=BASE_DIR)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--version", default="tmax_stage1_patch_pixelshuffle_10yr")
    p.add_argument("--data-version", default="tmax_stage1_patch_10yr_data")
    p.add_argument("--checkpoint", default="g_init.pth", choices=["g_init.pth", "g.pth"])
    p.add_argument("--var", default="tmax")
    p.add_argument("--year", type=int, default=1990)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-days", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--save-prediction", action="store_true")
    return p


def device_from_arg(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA/ROCm is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_field(path, variable, n_days=None):
    with Dataset(path) as ds:
        if variable not in ds.variables:
            raise KeyError(f"{variable!r} not found in {path}")
        values = ds.variables[variable][:n_days]
        values = np.asarray(np.ma.filled(values, 0.0), dtype=np.float32)
    np.nan_to_num(values, copy=False)
    return values


def scale(values, scaler):
    return scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)


def inverse_scale(values, scaler):
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape).astype(np.float32)


def load_elevation(base_dir, data_version):
    path = base_dir / "output" / data_version / "elev_lr_scaled.npy"
    elev = np.load(path, mmap_mode="r")
    return np.asarray(elev[0, :, :, 0], dtype=np.float32)


@torch.inference_mode()
def predict(model, lr, elevation, scaler, device, batch_size):
    predictions = []
    baselines = []
    for start in range(0, len(lr), batch_size):
        stop = min(start + batch_size, len(lr))
        lr_scaled = scale(lr[start:stop], scaler)
        elev = np.broadcast_to(elevation, lr_scaled.shape)
        inputs = np.stack((lr_scaled, elev), axis=1).copy()
        inputs = torch.from_numpy(inputs).to(device)
        pred_scaled = model(inputs)
        baseline_scaled = F.interpolate(
            inputs[:, :1], scale_factor=4, mode="bilinear", align_corners=False
        )
        predictions.append(pred_scaled[:, 0].cpu().numpy())
        baselines.append(baseline_scaled[:, 0].cpu().numpy())
    return (
        inverse_scale(np.concatenate(predictions), scaler),
        inverse_scale(np.concatenate(baselines), scaler),
    )


def metrics(prediction, truth):
    error = prediction - truth
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def clean_map(ax, values, title, cmap, vmin=None, vmax=None):
    image = ax.imshow(values[::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_temporal_means(plot_dir, lr, pred, truth):
    lr_mean = lr.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    truth_mean = truth.mean(axis=0)
    diff = pred_mean - truth_mean
    lo = float(min(np.percentile(pred_mean, 2), np.percentile(truth_mean, 2)))
    hi = float(max(np.percentile(pred_mean, 98), np.percentile(truth_mean, 98)))
    dmax = max(float(np.percentile(np.abs(diff), 98)), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    im0 = clean_map(axes[0], lr_mean, "Mean Daymet/ERA5 input 1°", "Spectral_r")
    clean_map(axes[1], pred_mean, "Mean prediction 0.25°", "Spectral_r", lo, hi)
    im2 = clean_map(axes[2], truth_mean, "Mean Daymet/ERA5 truth 0.25°", "Spectral_r", lo, hi)
    im3 = clean_map(axes[3], diff, "Mean prediction − truth", "RdBu_r", -dmax, dmax)
    fig.colorbar(im0, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.colorbar(im2, ax=axes[1:3], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.colorbar(im3, ax=axes[3], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.suptitle("Out-of-sample 1990 temporal means")
    fig.savefig(plot_dir / "temporal_mean_prediction_truth_difference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(plot_dir, pred, truth):
    error = pred - truth
    bias = error.mean(axis=0)
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt((error * error).mean(axis=0))
    dmax = max(float(np.percentile(np.abs(bias), 98)), 1e-6)
    emax = max(float(np.percentile(np.stack((mae, rmse)), 98)), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    im0 = clean_map(axes[0], bias, "Annual mean bias", "RdBu_r", -dmax, dmax)
    im1 = clean_map(axes[1], mae, "Daily MAE", "magma", 0, emax)
    clean_map(axes[2], rmse, "Daily RMSE", "magma", 0, emax)
    fig.colorbar(im0, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.colorbar(im1, ax=axes[1:], orientation="horizontal", fraction=0.05, pad=0.06)
    fig.savefig(plot_dir / "bias_mae_rmse_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_daily_metrics(plot_dir, pred, baseline, truth):
    pred_error = pred - truth
    base_error = baseline - truth
    pred_mae = np.mean(np.abs(pred_error), axis=(1, 2))
    pred_rmse = np.sqrt(np.mean(pred_error * pred_error, axis=(1, 2)))
    base_mae = np.mean(np.abs(base_error), axis=(1, 2))
    fig, ax = plt.subplots(figsize=(11, 5))
    days = np.arange(1, len(pred) + 1)
    ax.plot(days, pred_mae, label="SRGAN MAE", linewidth=1)
    ax.plot(days, pred_rmse, label="SRGAN RMSE", linewidth=1)
    ax.plot(days, base_mae, label="Bilinear MAE", linewidth=1, alpha=0.8)
    ax.set(xlabel="Day of 1990", ylabel="°C", title="Daily paired Daymet evaluation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "daily_metrics.png", dpi=220)
    plt.close(fig)


def main():
    args = parser().parse_args()
    device = device_from_arg(args.device)
    model_dir = args.base_dir / "models" / args.version
    output_dir = args.base_dir / "output" / args.version / f"daymet_{args.year}_evaluation"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    lr_path = args.data_dir / f"Daymet_ERA5_{args.var}_dy_{args.year}_1deg.nc"
    truth_path = args.data_dir / f"Daymet_ERA5_{args.var}_dy_{args.year}_0p25deg.nc"
    variable = f"{args.var}_dy"
    lr = read_field(lr_path, variable, args.n_days)
    truth = read_field(truth_path, variable, args.n_days)
    if len(lr) != len(truth):
        raise ValueError(f"Time dimensions differ: LR {len(lr)}, truth {len(truth)}")

    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = SRGAN_g_lr_patch(in_channels=2).to(device)
    model.load_state_dict(torch.load(model_dir / args.checkpoint, map_location=device))
    model.eval()
    elevation = load_elevation(args.base_dir, args.data_version)
    pred, baseline = predict(model, lr, elevation, scaler, device, args.batch_size)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction {pred.shape} and truth {truth.shape} differ")

    srgan_metrics = metrics(pred, truth)
    baseline_metrics = metrics(baseline, truth)
    mean_pred = pred.mean(axis=0)
    mean_truth = truth.mean(axis=0)
    mean_field_metrics = metrics(mean_pred, mean_truth)

    if args.save_prediction:
        np.save(output_dir / f"{args.var}_{args.year}_pred_0p25.npy", pred)
    np.save(output_dir / f"{args.var}_{args.year}_temporal_mean_pred.npy", mean_pred)
    np.save(output_dir / f"{args.var}_{args.year}_temporal_mean_truth.npy", mean_truth)

    plot_temporal_means(plot_dir, lr, pred, truth)
    plot_error_maps(plot_dir, pred, truth)
    plot_daily_metrics(plot_dir, pred, baseline, truth)

    summary = {
        "year": args.year,
        "variable": args.var,
        "checkpoint": args.checkpoint,
        "n_days": int(len(pred)),
        "lr_path": str(lr_path),
        "truth_path": str(truth_path),
        "prediction_shape": list(map(int, pred.shape)),
        "daily_all_grid_cells": srgan_metrics,
        "bilinear_baseline": baseline_metrics,
        "temporal_mean_field": mean_field_metrics,
        "mae_improvement_over_bilinear_percent": float(
            100 * (baseline_metrics["mae"] - srgan_metrics["mae"]) / baseline_metrics["mae"]
        ),
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
