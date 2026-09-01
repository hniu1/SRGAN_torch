#!/usr/bin/env python3
"""Prepare the lazy 0.25-degree to 1/24-degree multivariable Stage-2 index."""

from __future__ import annotations

import argparse
from pathlib import Path

from climate_downscaling.prepare import DEFAULT_DATA_ROOT, DEFAULT_DEM_ROOT
from climate_downscaling.stage2_prepare import prepare_stage2_index
from pipeline_01_prepare_multivariable import year_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/daymet_mv_stage2_1980_1990"))
    parser.add_argument("--stage1-manifest", type=Path, default=Path("artifacts/data/daymet_mv_1980_1990/manifest.json"))
    parser.add_argument("--variables", nargs="+", default=["tmin", "tmax", "prcp"])
    parser.add_argument("--train-start", type=int, default=1980)
    parser.add_argument("--train-end", type=int, default=1988)
    parser.add_argument("--val-start", type=int, default=1988)
    parser.add_argument("--val-end", type=int, default=1990)
    parser.add_argument("--test-start", type=int, default=1990)
    parser.add_argument("--test-end", type=int, default=1991)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dem-root", type=Path, default=DEFAULT_DEM_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = {
        "train": year_range(args.train_start, args.train_end),
        "val": year_range(args.val_start, args.val_end),
        "test": year_range(args.test_start, args.test_end),
    }
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Stage-2 chronological splits overlap")
    prepare_stage2_index(
        args.output_dir, args.variables, splits, args.stage1_manifest,
        data_root=args.data_root, dem_root=args.dem_root, scale_factor=6,
    )


if __name__ == "__main__":
    main()
