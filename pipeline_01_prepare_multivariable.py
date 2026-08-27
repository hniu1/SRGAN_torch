#!/usr/bin/env python3
"""Prepare aligned tmin/tmax/precipitation arrays for ClimateSwin."""

from __future__ import annotations

import argparse
from pathlib import Path

from climate_downscaling.prepare import DEFAULT_DATA_ROOT, DEFAULT_DEM_ROOT, prepare_dataset


def year_range(start: int, end: int) -> list[int]:
    if end <= start:
        raise argparse.ArgumentTypeError("year end must be greater than year start")
    return list(range(start, end))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/daymet_mv_1980_1990"))
    parser.add_argument("--variables", nargs="+", default=["tmin", "tmax", "prcp"])
    parser.add_argument("--train-start", type=int, default=1980)
    parser.add_argument("--train-end", type=int, default=1988, help="Exclusive")
    parser.add_argument("--val-start", type=int, default=1988)
    parser.add_argument("--val-end", type=int, default=1990, help="Exclusive")
    parser.add_argument("--test-start", type=int, default=1990)
    parser.add_argument("--test-end", type=int, default=1991, help="Exclusive")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dem-root", type=Path, default=DEFAULT_DEM_ROOT)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild only the variables named by --variables; shared arrays are retained",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = {
        "train": year_range(args.train_start, args.train_end),
        "val": year_range(args.val_start, args.val_end),
        "test": year_range(args.test_start, args.test_end),
    }
    overlap = (set(splits["train"]) & set(splits["val"])) | (set(splits["train"]) & set(splits["test"])) | (set(splits["val"]) & set(splits["test"]))
    if overlap:
        raise ValueError(f"Chronological splits overlap in years: {sorted(overlap)}")
    prepare_dataset(
        output_dir=args.output_dir,
        variables=args.variables,
        split_years=splits,
        data_root=args.data_root,
        dem_root=args.dem_root,
        scale_factor=args.scale_factor,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
