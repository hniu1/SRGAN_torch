#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Import your existing utilities
# ------------------------------------------------------------
# These should already exist in your codebase
from dataread import read_saved_data       # adjust import if needed
from dataread_mem import daymetread       # adjust import if needed

# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser("Prepare Daymet data (single-rank)")
    p.add_argument("--base-dir", default=os.environ.get("SRGAN_BASE_DIR", str(Path(__file__).resolve().parent)), help="Project base directory")
    p.add_argument("--version", default='dy_v0.5', help="Experiment version")
    p.add_argument("--var", default='tmax', help="Variable name (e.g., prcp, tmax)")
    p.add_argument("--year-start", type=int, default=1980)
    p.add_argument("--year-end", type=int, default=2020)
    p.add_argument("--high-deg", action="store_true", default=False,
                   help="Use high-resolution degree data (0.25 deg)")
    p.add_argument("--overwrite", action="store_true", default=True,
                   help="Overwrite existing cached data")
    return p


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    base_dir = Path(args.base_dir)
    version = args.version
    var = args.var.lower()
    high_deg = args.high_deg
    # high_deg = 1

    checkpoint_dir = base_dir / "models" / version
    path_output    = base_dir / "output" / version

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path_output.mkdir(parents=True, exist_ok=True)

    scaler_path = checkpoint_dir / "scaler.pkl"

    # --------------------------------------------------------
    # Skip if already prepared
    # --------------------------------------------------------
    if scaler_path.exists() and not args.overwrite:
        print("[prepare_daymet] Cached data already exists.")
        print("Scaler:", scaler_path)
        print("Use --overwrite to regenerate.")
        return

    print("=" * 80)
    print("[prepare_daymet] START")
    print(f"Base dir      : {base_dir}")
    print(f"Version       : {version}")
    print(f"Variable      : {var}")
    print(f"Years         : {args.year_start}–{args.year_end}")
    print("=" * 80, flush=True)

    # --------------------------------------------------------
    # Configuration (match training exactly)
    # --------------------------------------------------------
    elevation = True
    elevation_hr = False

    # --------------------------------------------------------
    # Generate data (THIS IS THE EXPENSIVE STEP)
    # --------------------------------------------------------
    try:
        _ = daymetread(
            path_output=str(path_output),
            checkpoint_dir=str(checkpoint_dir),
            elevation=elevation,
            elevation_hr=elevation_hr,
            Daymet_ERA5=True,
            high_deg=high_deg,
            scaler="minmax",
            var=var,
            year_start=args.year_start,
            year_end=args.year_end,
            yearly=True,
        )
    except Exception as e:
        print("[prepare_daymet] ERROR during daymetread()", flush=True)
        raise

    # --------------------------------------------------------
    # Validate outputs
    # --------------------------------------------------------
    required_files = [
        checkpoint_dir / "scaler.pkl",
        path_output / "x_train.npy",
        path_output / "x_test.npy",
        path_output / "y_train.npy",
        path_output / "y_test.npy",
    ]

    print("\n[prepare_daymet] Verifying outputs:")
    for f in required_files:
        if not f.exists():
            raise RuntimeError(f"Missing output file: {f}")
        print(f"  OK: {f}")

    # --------------------------------------------------------
    # Sanity check load
    # --------------------------------------------------------
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    x_train = read_saved_data("x_train", path_output, scaler)
    y_train = read_saved_data("y_train", path_output, scaler)

    print("\n[prepare_daymet] Sanity check:")
    print(f"  x_train shape: {x_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    print("\n[prepare_daymet] DONE SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
