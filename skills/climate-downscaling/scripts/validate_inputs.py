#!/usr/bin/env python3
"""Validate multivariable NetCDF inputs against a ClimateSwin stage contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("netCDF4 is required; use the repository Python environment") from exc


ALIASES = {
    "tmin": ("tmin",),
    "tmax": ("tmax",),
    "prcp": ("prcp", "pr", "precipitation"),
}
EXPECTED_SHAPES = {"stage1": (57, 129), "stage2": (228, 516)}
CELSIUS_UNITS = {
    "c", "degc", "degree_celsius", "degrees_celsius", "degree c", "degrees c", "celsius"
}
PRECIP_UNITS = {"mm/day", "mm/dy", "mm d-1", "mm day-1", "mm/day-1", "mm per day", "mm"}


def parse_mapping(values: list[str]) -> tuple[dict[str, Path], list[str]]:
    inputs: dict[str, Path] = {}
    errors: list[str] = []
    for value in values:
        if "=" not in value:
            errors.append(f"invalid input mapping {value!r}; expected VARIABLE=PATH")
            continue
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if name not in ALIASES:
            errors.append(f"unknown canonical variable {name!r}")
        elif name in inputs:
            errors.append(f"duplicate input for {name}")
        else:
            inputs[name] = Path(raw_path).expanduser()
    for required in ALIASES:
        if required not in inputs:
            errors.append(f"missing required input {required}")
    return inputs, errors


def resolve_variable(dataset: Dataset, canonical: str) -> str | None:
    for candidate in ALIASES[canonical]:
        if candidate in dataset.variables:
            return candidate
    return None


def normalized_units(value: Any) -> str:
    return str(value or "").strip().lower().replace("°", "deg")


def inspect_input(
    canonical: str,
    path: Path,
    expected_shape: tuple[int, int],
    strict: bool,
    full_scan: bool,
) -> tuple[dict[str, Any], list[str], list[str], np.ndarray | None]:
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"canonical_variable": canonical, "path": str(path)}
    if not path.is_file():
        errors.append(f"{canonical}: file not found: {path}")
        return result, errors, warnings, None

    try:
        with Dataset(path, "r") as dataset:
            source_name = resolve_variable(dataset, canonical)
            if source_name is None:
                errors.append(f"{canonical}: none of {ALIASES[canonical]} found in {path}")
                return result, errors, warnings, None
            variable = dataset.variables[source_name]
            result.update(
                source_variable=source_name,
                dimensions=list(variable.dimensions),
                shape=list(variable.shape),
                units=str(getattr(variable, "units", "")),
            )
            if variable.ndim != 3:
                errors.append(f"{canonical}: expected 3 dimensions, found {variable.ndim}")
                return result, errors, warnings, None
            if tuple(variable.shape[-2:]) != expected_shape:
                errors.append(
                    f"{canonical}: expected spatial shape {expected_shape}, found {tuple(variable.shape[-2:])}"
                )

            units = normalized_units(getattr(variable, "units", ""))
            accepted = CELSIUS_UNITS if canonical in ("tmin", "tmax") else PRECIP_UNITS
            if not units:
                message = f"{canonical}: units attribute is missing"
                (errors if strict else warnings).append(message)
            elif units not in accepted:
                message = f"{canonical}: units {units!r} do not match the required inference units"
                (errors if strict else warnings).append(message)

            sample_values = None
            if variable.shape[0] > 0:
                sample = np.ma.asarray(variable[0])
                values = np.asarray(sample.filled(np.nan), dtype=np.float64)
                sample_values = values
                finite = values[np.isfinite(values)]
                missing = int(np.count_nonzero(~np.isfinite(values)))
                minimum = float(finite.min()) if finite.size else None
                maximum = float(finite.max()) if finite.size else None
                scanned = 1
                if full_scan:
                    for start in range(1, variable.shape[0], 32):
                        chunk = np.ma.asarray(variable[start : start + 32])
                        chunk_values = np.asarray(chunk.filled(np.nan), dtype=np.float64)
                        missing += int(np.count_nonzero(~np.isfinite(chunk_values)))
                        chunk_finite = chunk_values[np.isfinite(chunk_values)]
                        if chunk_finite.size:
                            chunk_min = float(chunk_finite.min())
                            chunk_max = float(chunk_finite.max())
                            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
                            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
                        scanned += int(chunk_values.shape[0])
                scope = "all scanned timesteps" if full_scan else "first timestep"
                result["scanned_timesteps"] = scanned
                result["scanned_missing_cells"] = missing
                if minimum is not None:
                    result["scanned_min"] = minimum
                    result["scanned_max"] = maximum
                if missing:
                    message = f"{canonical}: {scope} contain {missing} missing/non-finite cells"
                    (errors if strict else warnings).append(message)
                if canonical == "prcp" and minimum is not None and minimum < 0:
                    errors.append(f"prcp: {scope} contain negative precipitation ({minimum:.6g})")
    except OSError as exc:
        errors.append(f"{canonical}: cannot open {path}: {exc}")
        sample_values = None
    return result, errors, warnings, sample_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(EXPECTED_SHAPES), required=True)
    parser.add_argument("--input", action="append", default=[], metavar="VARIABLE=PATH")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat missing/unknown units and sampled missing data as errors",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="scan every timestep for missing values and precipitation range in bounded chunks",
    )
    args = parser.parse_args()

    inputs, errors = parse_mapping(args.input)
    warnings: list[str] = []
    inspected: dict[str, Any] = {}
    time_lengths: dict[str, int] = {}
    samples: dict[str, np.ndarray] = {}
    for canonical, path in inputs.items():
        info, found_errors, found_warnings, sample = inspect_input(
            canonical, path, EXPECTED_SHAPES[args.stage], args.strict, args.full_scan
        )
        inspected[canonical] = info
        errors.extend(found_errors)
        warnings.extend(found_warnings)
        shape = info.get("shape")
        if shape and len(shape) == 3:
            time_lengths[canonical] = int(shape[0])
        if sample is not None:
            samples[canonical] = sample
    if len(set(time_lengths.values())) > 1:
        errors.append(f"time dimension lengths do not match: {time_lengths}")
    if "tmin" in samples and "tmax" in samples and samples["tmin"].shape == samples["tmax"].shape:
        finite = np.isfinite(samples["tmin"]) & np.isfinite(samples["tmax"])
        violations = int(np.count_nonzero(finite & (samples["tmin"] > samples["tmax"])))
        inspected["temperature_order_first_timestep_violations"] = violations
        if violations:
            warnings.append(f"first timestep contains {violations} cells where tmin > tmax")

    report = {
        "status": "valid" if not errors else "invalid",
        "stage": args.stage,
        "expected_spatial_shape": list(EXPECTED_SHAPES[args.stage]),
        "inputs": inspected,
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            (
                "temperature ordering is checked on the first timestep only"
                if args.full_scan
                else "only the first timestep is sampled for values and temperature ordering"
            ),
            "coordinate identity is not proven when coordinates are absent or stored separately",
        ],
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
