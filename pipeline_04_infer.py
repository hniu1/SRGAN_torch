#!/usr/bin/env python3
"""Apply a trained joint REFINE checkpoint to aligned coarse environmental fields."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch

from refine_downscaling.data import FullFieldDataset, validate_checkpoint_manifest
from refine_downscaling.model import REFINE, REFINEConfig
from refine_downscaling.transforms import (
    inverse_channels_numpy,
    specs_from_manifest,
    transform_channels_numpy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("daymet/prepared/stage1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/refine_stage1_5x/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", action="append", required=True, metavar="VARIABLE=PATH",
        help="Repeat once per model variable; prcp files may contain a variable named pr",
    )
    parser.add_argument("--start-date", help="Date of input index zero; CF time is used when available")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, help="Exclusive; defaults to all common timesteps")
    parser.add_argument("--batch-size", type=int, default=1)
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
    candidates = [model_name, f"{model_name}_dy"]
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
        dates: list[dt.datetime] | None = None,
        coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required for --format netcdf") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(path, "w", format="NETCDF4")
        self.dataset.createDimension("time", count)
        spatial_dims = ("lat", "lon") if coordinates is not None else ("y", "x")
        self.dataset.createDimension(spatial_dims[0], height)
        self.dataset.createDimension(spatial_dims[1], width)
        time = self.dataset.createVariable("time", "f8", ("time",))
        time.units = f"days since {start_date.isoformat()}"
        time.calendar = "proleptic_gregorian"
        if dates is None:
            time[:] = np.arange(count, dtype=np.float64)
        else:
            from netCDF4 import date2num

            time[:] = date2num(dates, time.units, time.calendar)
        for axis, size, dim in zip(range(2), (height, width), spatial_dims):
            coordinate = self.dataset.createVariable(dim, "f8" if coordinates is not None else "i4", (dim,))
            coordinate[:] = coordinates[axis] if coordinates is not None else np.arange(size)
            if coordinates is not None:
                coordinate.units = "degrees_north" if dim == "lat" else "degrees_east"
                coordinate.standard_name = "latitude" if dim == "lat" else "longitude"
        self.variables = {}
        for name in variable_names:
            variable = self.dataset.createVariable(
                name, "f4", ("time", *spatial_dims),
                zlib=True, complevel=2, shuffle=True,
                chunksizes=(1, min(height, 228), min(width, 516)),
                fill_value=np.float32(np.nan),
            )
            variable.long_name = f"REFINE downscaled {name}"
            if name in variable_metadata:
                variable.units = str(variable_metadata[name].get("units", ""))
                variable.training_long_name = str(variable_metadata[name].get("long_name", name))
            variable.source_file = str(source_paths[name])
            self.variables[name] = variable
        self.dataset.model = f"REFINE joint multivariable {scale_factor}x downscaler"

    def write(self, start: int, values: np.ndarray) -> None:
        for channel, (name, variable) in enumerate(self.variables.items()):
            variable[start:start + values.shape[0]] = values[:, channel]
        self.dataset.sync()

    def close(self) -> None:
        self.dataset.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    input_paths = parse_inputs(args.input)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    specs = specs_from_manifest(manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = REFINEConfig.from_dict(checkpoint["model_config"])
    validate_checkpoint_manifest(checkpoint, manifest)
    missing = set(config.variable_names) - set(input_paths)
    extra = set(input_paths) - set(config.variable_names)
    if missing or extra:
        raise ValueError(f"Input/model variable mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    start_date = dt.date.fromisoformat(args.start_date or "1990-01-01")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = REFINE(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if manifest.get("storage_layout") == "variable_separable_npy":
        static_source = FullFieldDataset(
            args.data_dir, split="test", variable_names=config.variable_names
        )
    elif manifest.get("storage_layout") == "netcdf_patch_index":
        from refine_downscaling.stage2_data import Stage2FullFieldDataset

        static_source = Stage2FullFieldDataset(
            args.data_dir, split="test", variable_names=config.variable_names
        )
    else:
        raise ValueError(f"Unsupported inference storage layout: {manifest.get('storage_layout')!r}")
    static_lr = torch.from_numpy(static_source.static_lr)[None].to(device)
    static_hr = torch.from_numpy(static_source.static_hr)[None].to(device)
    lr_shape = tuple(int(v) for v in manifest["lr_shape"])
    hr_shape = tuple(int(v) for v in manifest["hr_shape"])
    is_tva = manifest.get("source", {}).get("layout") == "daymet_tva"
    coordinates = None
    if is_tva:
        coordinates = tuple(np.load(args.data_dir / "shared" / f"{axis}_hr.npy") for axis in ("lat", "lon"))
        lr_coordinates = tuple(np.load(args.data_dir / "shared" / f"{axis}_lr.npy") for axis in ("lat", "lon"))

    with ExitStack() as stack:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required to read GCM inputs") from exc
        readers = {}
        source_names = {}
        alignment = {}
        lengths = []
        source_dates = None
        for name in config.variable_names:
            dataset = stack.enter_context(Dataset(input_paths[name]))
            variable, source_name = resolve_nc_variable(dataset, name)
            if variable.ndim != 3:
                raise ValueError(f"{name}: expected a three-dimensional time/spatial field")
            if is_tva:
                from refine_downscaling.daymet_prepare import check_grid, read_grid

                if variable.dimensions != ("time", "lat", "lon"):
                    raise ValueError(f"{name}: expected time,lat,lon dimensions")
                check_grid(read_grid(dataset), lr_coordinates, name)
            if "time" in dataset.variables and hasattr(dataset.variables["time"], "units"):
                from refine_downscaling.daymet_prepare import read_dates

                dates = read_dates(dataset)
                if len(dates) != variable.shape[0]:
                    raise ValueError(f"{name}: time coordinates and field lengths differ")
                if source_dates is not None and source_dates != dates:
                    raise ValueError(f"{name}: timestamps differ across inputs")
                source_dates = dates
            if tuple(variable.shape[-2:]) != lr_shape:
                raise ValueError(f"{name} grid {variable.shape[-2:]} does not match model LR grid {lr_shape}")
            from refine_downscaling.prepare import _canonical_units
            _, conversion = _canonical_units(name, str(getattr(variable, "units", "")))
            if conversion != "none":
                raise ValueError(f"{name}: inference requires Celsius, not Kelvin")
            for key in ("lat", "lon", "time"):
                if key in dataset.variables:
                    coordinate = dataset.variables[key]
                    values = np.asarray(coordinate[:])
                    attributes = (getattr(coordinate, "units", None), getattr(coordinate, "calendar", None))
                    if key in alignment:
                        reference, reference_attributes = alignment[key]
                        if not np.array_equal(reference, values) or attributes != reference_attributes:
                            raise ValueError(f"{name}: {key} coordinates differ across inputs")
                    else:
                        alignment[key] = (values, attributes)
            readers[name] = variable
            source_names[name] = source_name
            lengths.append(int(variable.shape[0]))
        if len(set(lengths)) != 1:
            raise ValueError(f"Input time lengths differ: {lengths}")
        available = lengths[0]
        if is_tva and source_dates is None:
            raise ValueError("TVA inference requires CF time coordinates")
        if source_dates is None:
            source_dates = [dt.datetime.combine(start_date + dt.timedelta(days=i), dt.time()) for i in range(available)]
        elif args.start_date and source_dates[0].date() != start_date:
            raise ValueError("--start-date disagrees with the first input CF timestamp")
        start_date = source_dates[0].date()
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
                source_dates[args.start_index].date(), input_paths,
                manifest.get("variable_metadata", {}),
                config.scale_factor,
                dates=source_dates[args.start_index:end], coordinates=coordinates,
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
                    if not is_tva and not np.isfinite(raw).all():
                        raise ValueError("Inference inputs contain missing/nonfinite values")
                    for channel, name in enumerate(config.variable_names):
                        if name == "prcp" and np.any(raw[:, channel] < 0):
                            raise ValueError("Inference precipitation must be nonnegative")
                    valid_lr = np.isfinite(raw).all(axis=1)
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    normalized = np.stack([
                        transform_channels_numpy(sample, config.variable_names, specs) for sample in raw
                    ])
                    normalized *= valid_lr[:, None]
                    batch = torch.from_numpy(normalized).to(device)
                    static_lr_batch = static_lr.expand(batch.shape[0], -1, -1, -1).clone()
                    static_lr_batch[:, -1] *= torch.from_numpy(valid_lr).to(device)
                    static_hr_batch = static_hr.expand(batch.shape[0], -1, -1, -1)
                    seasons = []
                    for index in range(source_start, source_end):
                        date = source_dates[index]
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
                    if is_tva:
                        physical[:, :, np.asarray(static_source.static_hr[-1] < 0.5)] = np.nan
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
        "start_date": source_dates[args.start_index].date().isoformat(),
        "end_date": source_dates[end - 1].date().isoformat(),
        "stage": manifest.get("stage"),
        "scale_factor": config.scale_factor,
        "temperature_order_enforced": args.enforce_temperature_order,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
