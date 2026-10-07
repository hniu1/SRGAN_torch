#!/usr/bin/env python3
"""Prepare lazy Daymet indexes for 025deg -> 005deg -> 001deg REFINE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from refine_downscaling.prepare import DEFAULT_DATA_ROOT, DEFAULT_DEM_ROOT
from refine_downscaling.stage2_prepare import prepare_stage2_index
from refine_downscaling.daymet_prepare import DEFAULT_TVA_ROOT, prepare_daymet_index

def year_range(start: int, end: int) -> list[int]:
    if end <= start:
        raise ValueError("year end must exceed start")
    return list(range(start, end))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("daymet_tva", "legacy"), default="daymet_tva")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, help="Default: daymet/prepared/stageN")
    parser.add_argument("--normalization-manifest", type=Path, help="Optional frozen transforms; default: fit on training LR data")
    parser.add_argument("--dem-lr", type=Path, help="DEM NetCDF on the stage input lat/lon grid")
    parser.add_argument("--dem-hr", type=Path, help="DEM NetCDF on the stage target lat/lon grid")
    parser.add_argument("--variables", nargs="+", default=["tmin", "tmax", "prcp"])
    parser.add_argument("--train-start", type=int, default=1982)
    parser.add_argument("--train-end", type=int, default=2011, help="Exclusive")
    parser.add_argument("--val-start", type=int, default=2011)
    parser.add_argument("--val-end", type=int, default=2016, help="Exclusive")
    parser.add_argument("--test-start", type=int, default=2016)
    parser.add_argument("--test-end", type=int, default=2021, help="Exclusive")
    parser.add_argument("--data-root", type=Path)
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
        raise ValueError("Chronological splits overlap")
    if args.layout == "daymet_tva":
        prepare_daymet_index(
            args.output_dir or Path(f"daymet/prepared/stage{args.stage}"), args.variables, splits,
            data_root=args.data_root or DEFAULT_TVA_ROOT, stage=args.stage,
            dem_lr=args.dem_lr, dem_hr=args.dem_hr,
            normalization_manifest_path=args.normalization_manifest,
        )
    else:
        prepare_stage2_index(
            args.output_dir or Path("daymet/prepared"), args.variables, splits,
            args.normalization_manifest or Path("daymet/normalization.json"),
            data_root=args.data_root or DEFAULT_DATA_ROOT, dem_root=args.dem_root, scale_factor=6,
        )


if __name__ == "__main__":
    main()
