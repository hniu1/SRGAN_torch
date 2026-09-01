#!/usr/bin/env python3
"""Apply a trained joint ClimateSwin checkpoint to aligned coarse climate fields."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch

from climate_downscaling.data import FullFieldDataset
from climate_downscaling.model import ClimateSwin, ClimateSwinConfig
from climate_downscaling.transforms import (
    inverse_channels_numpy,
    specs_from_manifest,
    transform_channels_numpy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data/daymet_mv_1980_1990"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/climateswin_v1/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", action="append", required=True, metavar="VARIABLE=PATH",
        help="Repeat once per model variable; prcp files may contain a variable named pr",
    )
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, help="Exclusive; defaults to all common timesteps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--format", choices=["netcdf", "npy"], default="netcdf")
    parser.add_argument("--enforce-temperature-order", action="store_true")
    return parser


def parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Input must use VARIABLE=PATH syntax: {value!r}")
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate input for {name}")
        result[name] = Path(raw_path)
    return result


def resolve_nc_variable(dataset, model_name: str):
    candidates = [model_name]
    if model_name in {"prcp", "precip", "precipitation"}:
        candidates.extend(["pr", "precipitation"])
    for candidate in candidates:
        if candidate in dataset.variables:
            return dataset.variables[candidate], candidate
    raise KeyError(f"None of {candidates} found in {dataset.filepath()}")


class NpyWriter:
    def __init__(self, path: Path, shape: tuple[int, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)

    def write(self, start: int, values: np.ndarray) -> None:
        self.array[start:start + values.shape[0]] = values
        self.array.flush()

    def close(self) -> None:
        self.array.flush()


class NetcdfWriter:
    def __init__(
        self,
        path: Path,
        variable_names: tuple[str, ...],
        count: int,
        height: int,
        width: int,
        start_date: dt.date,
        source_paths: dict[str, Path],
        variable_metadata: dict,
        scale_factor: int,
    ) -> None:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required for --format netcdf") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(path, "w", format="NETCDF4")
        self.dataset.createDimension("time", count)
        self.dataset.createDimension("y", height)
        self.dataset.createDimension("x", width)
        time = self.dataset.createVariable("time", "i4", ("time",))
        time[:] = np.arange(count, dtype=np.int32)
        time.units = f"days since {start_date.isoformat()}"
        time.calendar = "proleptic_gregorian"
        self.dataset.createVariable("y", "i4", ("y",))[:] = np.arange(height)
        self.dataset.createVariable("x", "i4", ("x",))[:] = np.arange(width)
        self.variables = {}
        for name in variable_names:
            variable = self.dataset.createVariable(
                name, "f4", ("time", "y", "x"),
                zlib=True, complevel=2, shuffle=True,
                chunksizes=(1, min(height, 228), min(width, 516)),
                fill_value=np.float32(np.nan),
            )
            variable.long_name = f"ClimateSwin downscaled {name}"
            if name in variable_metadata:
                variable.units = str(variable_metadata[name].get("units", ""))
                variable.training_long_name = str(variable_metadata[name].get("long_name", name))
            variable.source_file = str(source_paths[name])
            self.variables[name] = variable
        self.dataset.model = f"ClimateSwin joint multivariable {scale_factor}x downscaler"

    def write(self, start: int, values: np.ndarray) -> None:
        for channel, (name, variable) in enumerate(self.variables.items()):
            variable[start:start + values.shape[0]] = values[:, channel]
        self.dataset.sync()

    def close(self) -> None:
        self.dataset.close()


def main() -> None:
    args = build_parser().parse_args()
    input_paths = parse_inputs(args.input)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    specs = specs_from_manifest(manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ClimateSwinConfig.from_dict(checkpoint["model_config"])
    missing = set(config.variable_names) - set(input_paths)
    extra = set(input_paths) - set(config.variable_names)
    if missing or extra:
        raise ValueError(f"Input/model variable mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    start_date = dt.date.fromisoformat(args.start_date)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = ClimateSwin(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if manifest.get("storage_layout") == "variable_separable_npy":
        static_source = FullFieldDataset(
            args.data_dir, split="test", variable_names=config.variable_names
        )
    elif manifest.get("storage_layout") == "netcdf_patch_index":
        from climate_downscaling.stage2_data import Stage2FullFieldDataset

        static_source = Stage2FullFieldDataset(
            args.data_dir, split="test", variable_names=config.variable_names
        )
    else:
        raise ValueError(f"Unsupported inference storage layout: {manifest.get('storage_layout')!r}")
    static_lr = torch.from_numpy(static_source.static_lr)[None].to(device)
    static_hr = torch.from_numpy(static_source.static_hr)[None].to(device)
    lr_shape = tuple(int(v) for v in manifest["lr_shape"])
    hr_shape = tuple(int(v) for v in manifest["hr_shape"])

    with ExitStack() as stack:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required to read GCM inputs") from exc
        readers = {}
        source_names = {}
        lengths = []
        for name in config.variable_names:
            dataset = stack.enter_context(Dataset(input_paths[name]))
            variable, source_name = resolve_nc_variable(dataset, name)
            if tuple(variable.shape[-2:]) != lr_shape:
                raise ValueError(f"{name} grid {variable.shape[-2:]} does not match model LR grid {lr_shape}")
            readers[name] = variable
            source_names[name] = source_name
            lengths.append(int(variable.shape[0]))
        available = min(lengths)
        end = available if args.end_index is None else min(args.end_index, available)
        if args.start_index < 0 or end <= args.start_index:
            raise ValueError(f"Invalid index range [{args.start_index}, {end}) for {available} samples")
        count = end - args.start_index

        output_shape = (count, len(config.variable_names), *hr_shape)
        if args.format == "npy":
            writer = NpyWriter(args.output, output_shape)
        else:
            writer = NetcdfWriter(
                args.output, config.variable_names, count, *hr_shape,
                start_date + dt.timedelta(days=args.start_index), input_paths,
                manifest.get("variable_metadata", {}),
                config.scale_factor,
            )
        try:
            output_index = 0
            with torch.no_grad():
                for source_start in range(args.start_index, end, args.batch_size):
                    source_end = min(source_start + args.batch_size, end)
                    raw = np.stack([
                        np.asarray(
                            np.ma.filled(readers[name][source_start:source_end], np.nan),
                            dtype=np.float32,
                        )
                        for name in config.variable_names
                    ], axis=1)
                    raw = np.nan_to_num(raw, nan=0.0)
                    normalized = np.stack([
                        transform_channels_numpy(sample, config.variable_names, specs) for sample in raw
                    ])
                    batch = torch.from_numpy(normalized).to(device)
                    static_lr_batch = static_lr.expand(batch.shape[0], -1, -1, -1)
                    static_hr_batch = static_hr.expand(batch.shape[0], -1, -1, -1)
                    seasons = []
                    for index in range(source_start, source_end):
                        date = start_date + dt.timedelta(days=index)
                        phase = 2.0 * np.pi * (date.timetuple().tm_yday - 1.0) / 365.25
                        seasons.append((np.sin(phase), np.cos(phase)))
                    season = torch.tensor(seasons, dtype=torch.float32, device=device)
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if args.amp and device.type == "cuda" else nullcontext()
                    )
                    with autocast:
                        predicted = model(batch, static_lr_batch, static_hr_batch, season)
                    predicted = predicted.float().cpu().numpy()
                    physical = np.stack([
                        inverse_channels_numpy(sample, config.variable_names, specs) for sample in predicted
                    ])
                    if args.enforce_temperature_order and {"tmin", "tmax"}.issubset(config.variable_names):
                        i_min = config.variable_names.index("tmin")
                        i_max = config.variable_names.index("tmax")
                        invalid = physical[:, i_min] > physical[:, i_max]
                        midpoint = 0.5 * (physical[:, i_min] + physical[:, i_max])
                        physical[:, i_min] = np.where(invalid, midpoint, physical[:, i_min])
                        physical[:, i_max] = np.where(invalid, midpoint, physical[:, i_max])
                    writer.write(output_index, physical)
                    output_index += physical.shape[0]
                    print(f"Downscaled {output_index}/{count} timesteps", flush=True)
        finally:
            writer.close()

    metadata = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "output": str(args.output),
        "format": args.format,
        "variables": list(config.variable_names),
        "variable_metadata": manifest.get("variable_metadata", {}),
        "source_variable_names": source_names,
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "source_start_index": args.start_index,
        "source_end_index": end,
        "output_timesteps": count,
        "start_date": (start_date + dt.timedelta(days=args.start_index)).isoformat(),
        "temperature_order_enforced": args.enforce_temperature_order,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
