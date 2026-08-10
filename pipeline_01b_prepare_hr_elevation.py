#!/usr/bin/env python3
"""Add a static 0.25-degree elevation predictor to an existing data cache."""

import argparse
from pathlib import Path

import numpy as np

from dataread_mem import _read_elev_2d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--data-version", default="tmax_stage1_patch_10yr_data")
    args = p.parse_args()

    output_dir = args.base_dir / "output" / args.data_version
    if not (output_dir / "x_train.npy").exists():
        raise FileNotFoundError(f"Prepared data cache not found: {output_dir}")

    elevation_lr = np.squeeze(_read_elev_2d(1)).astype(np.float32)
    elevation_hr = np.squeeze(_read_elev_2d(0.25)).astype(np.float32)
    lr_min = float(np.min(elevation_lr))
    lr_max = float(np.max(elevation_lr))
    scale = max(lr_max - lr_min, 1e-12)
    elevation_hr_scaled = ((elevation_hr - lr_min) / scale).astype(np.float32)

    expected = tuple(v * 4 for v in elevation_lr.shape)
    if elevation_hr_scaled.shape != expected:
        raise ValueError(
            f"HR elevation shape {elevation_hr_scaled.shape}; expected {expected}"
        )
    path = output_dir / "elev_hr_scaled.npy"
    np.save(path, elevation_hr_scaled)
    print(f"Wrote {path}")
    print(f"shape={elevation_hr_scaled.shape}, range="
          f"({elevation_hr_scaled.min():.6f}, {elevation_hr_scaled.max():.6f})")


if __name__ == "__main__":
    main()
