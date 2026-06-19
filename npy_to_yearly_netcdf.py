#!/usr/bin/env python3

"""
Convert pre-computed y_pred_4.npy (all-years downscaled data) directly to yearly NetCDF files.

Skips GPU inference entirely. Useful if you already have the full downscaled array and just need
to split it into yearly files with proper dimensions.
"""

import argparse
import calendar
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DAYMET_DATA_DIR = "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data"


def build_parser():
    p = argparse.ArgumentParser(description="Convert y_pred_4.npy to yearly NetCDF files")

    p.add_argument("--var", required=True, choices=["pr", "tmax", "tmin"])
    p.add_argument("--gcm", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--ensemble", default="r1i1p1f1")
    p.add_argument("--grid", default="gn")

    p.add_argument("--downscale-version", required=True, help="Version tag (e.g. 0.2)")
    p.add_argument("--year-start", type=int, required=True, help="Inclusive year start")
    p.add_argument("--year-end", type=int, required=True, help="Exclusive year end")

    p.add_argument("--npy-file", required=True, help="Path to y_pred_4.npy file")
    p.add_argument("--output-dir", required=True, help="Directory for yearly NetCDF outputs")
    p.add_argument("--daymet-data-dir", default=DAYMET_DATA_DIR)

    p.add_argument(
        "--file-pattern",
        choices=["legacy", "srcnn"],
        default="legacy",
        help="legacy: var_GCM_scenario_0.0416deg_predict_daily_SRGAN_year.nc; srcnn: var_day_GCM_scenario_ens_grid_year_SRGAN_version_00416deg.nc",
    )
    p.add_argument("--variable-name", default=None, help="Output NetCDF variable name (default: <var>_downscaled)")

    return p


def _find_daymet_template(daymet_data_dir, var, year):
    candidates = []
    if var == "pr":
        candidates.extend(
            [
                ("prcp_dy", Path(daymet_data_dir) / f"Daymet_ERA5_prcp_dy_{year}_trim.nc"),
                ("prcp_dy", Path(daymet_data_dir) / f"Daymet_ERA5_prcp_dy_{year}$_trim.nc"),
            ]
        )
    candidates.append((f"{var}_dy", Path(daymet_data_dir) / f"Daymet_ERA5_{var}_dy_{year}_trim.nc"))

    for var_name, file_path in candidates:
        if file_path.exists():
            return file_path, var_name

    raise FileNotFoundError(
        f"Could not find Daymet trim template for var={var}, year={year} under {daymet_data_dir}"
    )


def _read_daymet_grid(daymet_data_dir, var, year):
    template_path, template_var_name = _find_daymet_template(daymet_data_dir, var, year)
    with Dataset(template_path, "r") as ds:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        lat_dtype = ds.variables["lat"].dtype
        lon_dtype = ds.variables["lon"].dtype
        lat_attrs = {a: ds.variables["lat"].getncattr(a) for a in ds.variables["lat"].ncattrs()}
        lon_attrs = {a: ds.variables["lon"].getncattr(a) for a in ds.variables["lon"].ncattrs()}
        var_attrs = {}
        if template_var_name in ds.variables:
            var_attrs = {
                a: ds.variables[template_var_name].getncattr(a)
                for a in ds.variables[template_var_name].ncattrs()
            }

    return lat, lon, lat_dtype, lon_dtype, lat_attrs, lon_attrs, var_attrs


def _year_lengths(year_start, year_end):
    return [366 if calendar.isleap(y) else 365 for y in range(year_start, year_end)]


def _build_output_file(output_dir, pattern, var, gcm, scenario, ensemble, grid, year, downscale_version):
    if pattern == "srcnn":
        name = f"{var}_day_{gcm}_{scenario}_{ensemble}_{grid}_{year}_SRGAN_{downscale_version}_00416deg.nc"
    else:
        name = f"{var}_{gcm}_{scenario}_0.0416deg_predict_daily_SRGAN_{year}.nc"
    return Path(output_dir) / name


def _create_yearly_netcdf(path, year, days, lat, lon, lat_dtype, lon_dtype, lat_attrs, lon_attrs, out_var, out_units):
    ds = Dataset(path, "w", format="NETCDF4")

    ds.createDimension("time", days)
    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))

    t_out = ds.createVariable("time", "f8", ("time",))
    lat_out = ds.createVariable("lat", lat_dtype, ("lat",))
    lon_out = ds.createVariable("lon", lon_dtype, ("lon",))

    y_out = ds.createVariable(
        out_var,
        "f4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=4,
        fill_value=np.nan,
    )

    t_out[:] = np.arange(days, dtype=np.float64)
    t_out.units = f"days since {year}-01-01"
    t_out.calendar = "standard"

    lat_out[:] = lat
    lon_out[:] = lon

    for k, v in lat_attrs.items():
        lat_out.setncattr(k, v)
    for k, v in lon_attrs.items():
        lon_out.setncattr(k, v)

    y_out.units = out_units
    y_out.long_name = f"SRGAN downscaled {out_var}"

    ds.description = "SRGAN downscaled GCM output (yearly)"
    return ds, y_out


def run(args):
    npy_file = Path(args.npy_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not npy_file.exists():
        raise FileNotFoundError(f"Input npy file not found: {npy_file}")

    print(f"[info] loading {npy_file}")
    y_pred_4 = np.load(npy_file, mmap_mode="r")
    t_total, h4, w4 = y_pred_4.shape
    print(f"[info] array shape: {y_pred_4.shape}")

    year_days = _year_lengths(args.year_start, args.year_end)
    expected_total_days = sum(year_days)

    if t_total != expected_total_days:
        raise ValueError(
            f"Time length mismatch: npy has {t_total} days, but year range "
            f"{args.year_start}-{args.year_end} expects {expected_total_days} days."
        )

    lat, lon, lat_dtype, lon_dtype, lat_attrs, lon_attrs, var_attrs = _read_daymet_grid(
        args.daymet_data_dir, args.var, args.year_start
    )

    if (h4, w4) != (len(lat), len(lon)):
        raise ValueError(
            f"NPY grid {(h4, w4)} does not match Daymet grid {(len(lat), len(lon))}"
        )

    out_var = args.variable_name if args.variable_name else f"{args.var}_downscaled"
    default_units = var_attrs.get("units", "unknown")

    year_files = []
    for year in range(args.year_start, args.year_end):
        year_files.append(
            _build_output_file(
                output_dir,
                args.file_pattern,
                args.var,
                args.gcm,
                args.scenario,
                args.ensemble,
                args.grid,
                year,
                args.downscale_version,
            )
        )

    all_exist = all(p.exists() for p in year_files)
    if all_exist:
        print("[info] all yearly output files already exist; skip write")
        return

    print("[stage] writing yearly NetCDF files")

    day_offset = 0
    for year_idx, year in enumerate(range(args.year_start, args.year_end)):
        days = year_days[year_idx]
        year_file = year_files[year_idx]
        year_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"  writing {year} ({days} days)...")

        ds, var_out = _create_yearly_netcdf(
            year_file,
            year,
            days,
            lat,
            lon,
            lat_dtype,
            lon_dtype,
            lat_attrs,
            lon_attrs,
            out_var,
            default_units,
        )

        # Load and write year data
        year_data = np.asarray(y_pred_4[day_offset : day_offset + days], dtype=np.float32)
        var_out[:, :, :] = year_data

        ds.close()
        day_offset += days

    print("[done] yearly NetCDF files created")
    for p in year_files:
        print(f"  - {p}")


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
