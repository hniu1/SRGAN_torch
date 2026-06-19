#!/usr/bin/env python3
"""
Plot one-year PR downscaling comparison for each GCM with three rows:
1) Raw 100 km input
2) Old version (gcm_ds/0.1)
3) New version (gcm_ds/0.1_test_param)

Each figure has three columns (mean, p95, wetdays) and shared color scales
per metric across rows for fair visual comparison.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-row PR comparison plot per GCM")
    p.add_argument("--base-dir", default=".", help="Repository root path")
    p.add_argument("--var", default="pr", help="Variable name (default: pr)")
    p.add_argument("--old-version", default="0.1", help="Old downscaling version")
    p.add_argument("--new-version", default="0.1_test_param", help="New downscaling version")
    p.add_argument("--year-index", type=int, default=0, help="0-based year index to plot")
    p.add_argument("--days-per-year", type=int, default=365, help="Days per year window")
    p.add_argument("--wetday-threshold", type=float, default=1.0, help="Wetday threshold in mm/day")
    return p.parse_args()


def safe_percentile(arr: np.ndarray, q: float) -> float:
    v = np.nanpercentile(arr, q)
    if not np.isfinite(v):
        return 0.0
    return float(v)


def metric_maps(y: np.ndarray, wetday_threshold: float) -> Dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(y, axis=0),
        "p95": np.nanpercentile(y, 95, axis=0),
        "wetdays": np.mean(y > wetday_threshold, axis=0) * y.shape[0],
    }


def pick_raw_100km_path(base_dir: str, old_version: str, new_version: str, var: str, gcm: str) -> Optional[str]:
    cands = [
        os.path.join(base_dir, "gcm_ds", new_version, var, gcm, "y_gcm_100.npy"),
        os.path.join(base_dir, "gcm_ds", old_version, var, gcm, "y_gcm_100.npy"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def year_slice(arr: np.ndarray, year_index: int, days_per_year: int) -> np.ndarray:
    start = year_index * days_per_year
    end = start + days_per_year
    if start >= arr.shape[0]:
        raise ValueError(f"Requested year_index={year_index} exceeds available time steps {arr.shape[0]}")
    end = min(end, arr.shape[0])
    return arr[start:end]


def gcm_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    names = []
    for n in sorted(os.listdir(path)):
        full = os.path.join(path, n)
        if os.path.isdir(full):
            names.append(n)
    return names


def plot_one_gcm(
    gcm: str,
    y100: np.ndarray,
    yold: np.ndarray,
    ynew: np.ndarray,
    out_png: str,
    wetday_threshold: float,
    year_index: int,
) -> None:
    m100 = metric_maps(y100, wetday_threshold)
    mold = metric_maps(yold, wetday_threshold)
    mnew = metric_maps(ynew, wetday_threshold)

    vmax = {
        "mean": max(
            safe_percentile(m100["mean"], 99),
            safe_percentile(mold["mean"], 99),
            safe_percentile(mnew["mean"], 99),
        ),
        "p95": max(
            safe_percentile(m100["p95"], 99),
            safe_percentile(mold["p95"], 99),
            safe_percentile(mnew["p95"], 99),
        ),
        "wetdays": max(
            safe_percentile(m100["wetdays"], 99),
            safe_percentile(mold["wetdays"], 99),
            safe_percentile(mnew["wetdays"], 99),
        ),
    }

    fig, axs = plt.subplots(3, 3, figsize=(16, 12))

    rows: List[Tuple[str, Dict[str, np.ndarray]]] = [
        ("Raw input 100km", m100),
        ("Old version 0.1", mold),
        ("New version 0.1_test_param", mnew),
    ]
    cols = [
        ("mean", "Mean PR", "mm/day"),
        ("p95", "P95 PR", "mm/day"),
        ("wetdays", f"Wetdays > {wetday_threshold:g} mm/day", "days/year"),
    ]

    for r, (row_name, maps) in enumerate(rows):
        for c, (key, col_title, cbar_label) in enumerate(cols):
            ax = axs[r, c]
            im = ax.imshow(maps[key], origin="lower", cmap="Spectral", vmin=0, vmax=vmax[key])
            ax.set_xlabel("lon index")
            ax.set_ylabel("lat index")
            if r == 0:
                ax.set_title(col_title, fontsize=12)
            if c == 0:
                ax.text(
                    -0.07,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                )
            cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.9)
            cb.set_label(cbar_label)

    fig.suptitle(f"PR Pred25 comparison | {gcm} | year_index={year_index}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    base_dir = os.path.abspath(args.base_dir)
    old_root = os.path.join(base_dir, "gcm_ds", args.old_version, args.var)
    new_root = os.path.join(base_dir, "gcm_ds", args.new_version, args.var)

    gcms = gcm_dirs(new_root)
    if not gcms:
        raise FileNotFoundError(f"No GCM directories found under {new_root}")

    print(f"Found {len(gcms)} GCM directories under {new_root}")

    for gcm in gcms:
        old_pred_path = os.path.join(old_root, gcm, "y_pred_25.npy")
        new_pred_path = os.path.join(new_root, gcm, "y_pred_25.npy")
        raw100_path = pick_raw_100km_path(base_dir, args.old_version, args.new_version, args.var, gcm)

        if raw100_path is None:
            print(f"[SKIP] {gcm}: missing y_gcm_100.npy in old/new version directories")
            continue
        if not os.path.exists(old_pred_path):
            print(f"[SKIP] {gcm}: missing old pred25: {old_pred_path}")
            continue
        if not os.path.exists(new_pred_path):
            print(f"[SKIP] {gcm}: missing new pred25: {new_pred_path}")
            continue

        y100 = np.load(raw100_path, mmap_mode="r")
        yold = np.load(old_pred_path, mmap_mode="r")
        ynew = np.load(new_pred_path, mmap_mode="r")

        y100_yr = year_slice(y100, args.year_index, args.days_per_year)
        yold_yr = year_slice(yold, args.year_index, args.days_per_year)
        ynew_yr = year_slice(ynew, args.year_index, args.days_per_year)

        out_dir = os.path.join(new_root, gcm, "fig")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"quick_pred25_3row_compare_year{args.year_index + 1}.png")

        plot_one_gcm(
            gcm=gcm,
            y100=y100_yr,
            yold=yold_yr,
            ynew=ynew_yr,
            out_png=out_png,
            wetday_threshold=args.wetday_threshold,
            year_index=args.year_index,
        )

        print(f"[OK] {gcm}: {out_png}")


if __name__ == "__main__":
    main()
