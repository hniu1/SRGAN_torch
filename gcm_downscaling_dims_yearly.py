#!/usr/bin/env python3

"""
SRGAN GCM downscaling with decoupled inference and yearly NetCDF output.

Three modes (--mode):
    downscale  : GPU inference only -> writes y_gcm_100.npy, y_pred_25.npy, y_pred_4.npy
    save-nc    : CPU only -> reads y_pred_4.npy and writes yearly NetCDF with time/lat/lon dims
    all        : runs downscale then save-nc in one job (default)
"""

import argparse
import calendar
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset
from tqdm import tqdm

from srgan_torch import SRGAN_g_hr_26_64RB, SRGAN_g_lr_26


DAYMET_DATA_DIR = "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data"


def build_parser():
    p = argparse.ArgumentParser(description="One-pass SRGAN yearly GCM inference")

    p.add_argument(
        "--mode",
        choices=["downscale", "save-nc", "all"],
        default="all",
        help=(
            "downscale: run GPU inference only (saves npy files); "
            "save-nc: convert existing y_pred_4.npy to yearly NetCDF (no GPU needed); "
            "all: run both stages (default)"
        ),
    )

    p.add_argument("--var", required=True, choices=["pr", "tmax", "tmin"])
    p.add_argument("--gcm", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--ensemble", default="r1i1p1f1")
    p.add_argument("--grid", default="gn")

    p.add_argument("--version-lr", required=True, help="SRGAN LR model version (e.g. dy_v0.3)")
    p.add_argument("--version-hr", required=True, help="SRGAN HR model version (e.g. dy_v0.4)")
    p.add_argument("--downscale-version", required=True, help="Output tag (e.g. 0.2)")

    p.add_argument("--year-start", type=int, required=True, help="Inclusive year start")
    p.add_argument("--year-end", type=int, required=True, help="Exclusive year end")

    p.add_argument("--gcm-file", default=None, help="Input combined 1deg GCM NetCDF file (required for downscale/all)")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output-dir", required=True, help="Directory for yearly NetCDF outputs")
    p.add_argument("--daymet-data-dir", default=DAYMET_DATA_DIR)

    p.add_argument("--elevation", action="store_true", help="Use elevation channel for both models")
    p.add_argument("--batch-size-lr", type=int, default=4)
    p.add_argument("--batch-size-hr", type=int, default=2)

    p.add_argument(
        "--file-pattern",
        choices=["legacy", "srcnn"],
        default="legacy",
        help="legacy: var_GCM_scenario_0.0416deg_predict_daily_SRGAN_year.nc; srcnn: var_day_GCM_scenario_ens_grid_year_SRGAN_version_00416deg.nc",
    )
    p.add_argument("--variable-name", default=None, help="Output NetCDF variable name (default: <var>_downscaled)")

    p.add_argument("--skip-y25", action="store_true", help="Delete y_pred_25.npy after 25->4 inference (default: keep)")
    p.add_argument("--skip-y4", action="store_true", help="Delete y_pred_4.npy after save-nc stage (default: keep)")
    p.add_argument(
        "--nc-compression-level",
        type=int,
        default=1,
        help="NetCDF compression level for yearly output (0 disables compression, default: 1)",
    )

    return p


def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict


def _load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


def _normalize_elevation(elev):
    arr = np.asarray(elev)
    if arr.ndim == 4:
        if arr.shape[0] > 1:
            arr = arr[0]
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr[:, :, 0]
    elif arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported elevation shape: {arr.shape}")
    return arr.astype(np.float32)


def _scale_batch_3d(batch_3d, scaler):
    batch_3d = np.asarray(batch_3d, dtype=np.float32)

    # Fast path for StandardScaler-like objects with scalar mean/scale.
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.float32(np.ravel(scaler.mean_)[0])
        scale = np.float32(np.ravel(scaler.scale_)[0])
        return ((batch_3d - mean) / scale).astype(np.float32, copy=False)

    shp = batch_3d.shape
    flat = batch_3d.reshape(-1, 1)
    scaled = scaler.transform(flat)
    return scaled.reshape(shp).astype(np.float32)


def _inverse_batch_3d(batch_3d, scaler):
    batch_3d = np.asarray(batch_3d, dtype=np.float32)

    # Fast path for StandardScaler-like objects with scalar mean/scale.
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.float32(np.ravel(scaler.mean_)[0])
        scale = np.float32(np.ravel(scaler.scale_)[0])
        return (batch_3d * scale + mean).astype(np.float32, copy=False)

    shp = batch_3d.shape
    flat = batch_3d.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(shp).astype(np.float32)


def _year_lengths(year_start, year_end):
    return [366 if calendar.isleap(y) else 365 for y in range(year_start, year_end)]


def _days_between_years(year_start, year_end):
    return sum(366 if calendar.isleap(y) else 365 for y in range(year_start, year_end))


def _parse_year_span_from_filename(path):
    """
    Parse year span from filename tokens such as:
    *_198001-201912_*  -> (1980, 2020)  # end is exclusive
    *_202001-205912_*  -> (2020, 2060)
    """
    m = re.search(r"_(\d{4})(\d{2})-(\d{4})(\d{2})_", Path(path).name)
    if not m:
        return None

    y0, m0, y1, m1 = map(int, m.groups())
    start_year = y0
    end_year_excl = y1 + 1 if m1 == 12 else y1

    # Only support yearly ranges that start in January for now.
    if m0 != 1:
        return None

    return start_year, end_year_excl


def _build_output_file(output_dir, pattern, var, gcm, scenario, ensemble, grid, year, downscale_version):
    if pattern == "srcnn":
        name = f"{var}_day_{gcm}_{scenario}_{ensemble}_{grid}_{year}_SRGAN_{downscale_version}_00416deg.nc"
    else:
        name = f"{var}_{gcm}_{scenario}_0.0416deg_predict_daily_SRGAN_{year}.nc"
    return Path(output_dir) / name


def _create_yearly_netcdf(
    path,
    year,
    days,
    lat,
    lon,
    lat_dtype,
    lon_dtype,
    lat_attrs,
    lon_attrs,
    out_var,
    out_units,
    nc_compression_level,
):
    ds = Dataset(path, "w", format="NETCDF4")

    ds.createDimension("time", days)
    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))

    t_out = ds.createVariable("time", "f8", ("time",))
    lat_out = ds.createVariable("lat", lat_dtype, ("lat",))
    lon_out = ds.createVariable("lon", lon_dtype, ("lon",))

    use_compression = nc_compression_level > 0
    y_out = ds.createVariable(
        out_var,
        "f4",
        ("time", "lat", "lon"),
        zlib=use_compression,
        complevel=nc_compression_level if use_compression else 0,
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


def _load_gcm_slice(gcm_file, var, year_start, year_end, expected_total_days):
    with Dataset(gcm_file, "r") as ds_gcm:
        if var not in ds_gcm.variables:
            raise KeyError(f"Variable '{var}' not found in {gcm_file}")
        gcm = ds_gcm.variables[var][:].astype(np.float32)

    t_total_raw = gcm.shape[0]
    if t_total_raw == expected_total_days:
        return gcm

    span = _parse_year_span_from_filename(gcm_file)
    if span is None:
        raise ValueError(
            f"Time length mismatch: gcm has {t_total_raw} days, expected {expected_total_days}. "
            "Could not infer span from filename; expected pattern like *_YYYY01-YYYY12_*."
        )

    file_start_year, file_end_year = span
    if year_start < file_start_year or year_end > file_end_year:
        raise ValueError(
            f"Requested years {year_start}-{year_end} are outside input span "
            f"{file_start_year}-{file_end_year}."
        )

    slice_start = _days_between_years(file_start_year, year_start)
    slice_end = _days_between_years(file_start_year, year_end)
    if slice_end > t_total_raw:
        raise ValueError(f"Computed slice [{slice_start}:{slice_end}] exceeds input length {t_total_raw}.")

    gcm = gcm[slice_start:slice_end]
    if gcm.shape[0] != expected_total_days:
        raise ValueError(f"Sliced input has {gcm.shape[0]} days, expected {expected_total_days}.")

    print(
        f"[info] extracted years {year_start}-{year_end} from span "
        f"{file_start_year}-{file_end_year} using slice [{slice_start}:{slice_end}]"
    )
    return gcm


def stage_downscale(args, npy_dir, year_days):
    """GPU stage: 100->25->4 km, writes npy files. Mirrors old gcm_downscaling.py behaviour."""
    base_dir = Path(args.base_dir).resolve()
    checkpoint_lr = base_dir / "models" / args.version_lr
    checkpoint_hr = base_dir / "models" / args.version_hr
    path_output_lr = base_dir / "output" / args.version_lr
    path_output_hr = base_dir / "output" / args.version_hr

    scaler_lr = _load_scaler(checkpoint_lr / "scaler.pkl")
    scaler_hr = _load_scaler(checkpoint_hr / "scaler.pkl")

    y100_path = npy_dir / "y_gcm_100.npy"
    y25_path  = npy_dir / "y_pred_25.npy"
    y4_path   = npy_dir / "y_pred_4.npy"

    t_total = sum(year_days)

    # ── 100→25 km ───────────────────────────────────────────────────────────
    if y25_path.exists():
        print(f"[skip] y_pred_25.npy exists at {y25_path}")
    else:
        if args.gcm_file is None:
            raise ValueError("--gcm-file is required for downscale mode")
        gcm = _load_gcm_slice(args.gcm_file, args.var, args.year_start, args.year_end, t_total)
        h100, w100 = gcm.shape[1], gcm.shape[2]
        print(f"[info] GCM shape = {gcm.shape}")

        # Save 100 km source
        np.save(y100_path, gcm.astype(np.float32))
        print(f"[info] saved {y100_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[info] device = {device}")

        g_lr = SRGAN_g_lr_26(in_channels=2 if args.elevation else 1).to(device)
        state = torch.load(checkpoint_lr / "g.pth", map_location=device)
        g_lr.load_state_dict(_strip_module_prefix(state))
        g_lr.eval()

        elev_lr = None
        if args.elevation:
            elev_lr = _normalize_elevation(np.load(path_output_lr / "elev_lr_scaled.npy"))
            if elev_lr.shape != (h100, w100):
                raise ValueError(f"LR elevation shape {elev_lr.shape} != GCM shape {(h100, w100)}")

        h25 = w25 = None
        y25_mm = None
        with torch.no_grad():
            for i in tqdm(range(0, t_total, args.batch_size_lr), desc="Infer 100->25"):
                j = min(i + args.batch_size_lr, t_total)
                batch_scaled = _scale_batch_3d(gcm[i:j], scaler_lr)
                if args.elevation:
                    x = np.stack([batch_scaled, np.broadcast_to(elev_lr, batch_scaled.shape)], axis=1)
                else:
                    x = batch_scaled[:, None, :, :]
                pred = _inverse_batch_3d(
                    g_lr(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()[:, 0, :, :],
                    scaler_lr,
                )
                if args.var == "pr":
                    np.maximum(pred, 0, out=pred)
                if y25_mm is None:
                    h25, w25 = pred.shape[1], pred.shape[2]
                    y25_mm = np.memmap(y25_path, dtype=np.float32, mode="w+", shape=(t_total, h25, w25))
                y25_mm[i:j] = pred
        y25_mm.flush()
        del y25_mm, gcm
        print(f"[done] 100->25 km saved to {y25_path}")

    # ── 25→4 km ─────────────────────────────────────────────────────────────
    if y4_path.exists():
        print(f"[skip] y_pred_4.npy exists at {y4_path}")
        return

    y25_mm = np.memmap(y25_path, dtype=np.float32, mode="r")
    # Infer shape from file size
    n_elements = y25_mm.size
    # We need h25/w25: load one batch to discover
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_hr = SRGAN_g_hr_26_64RB(in_channels=2 if args.elevation else 1).to(device)
    state = torch.load(checkpoint_hr / "g.pth", map_location=device)
    g_hr.load_state_dict(_strip_module_prefix(state))
    g_hr.eval()

    # Reload y25 with correct shape
    del y25_mm
    # Discover h25/w25 from element count
    n_el = y25_path.stat().st_size // 4  # float32
    hw = n_el // t_total
    # h25*w25 = hw; pick reasonable aspect from 4x upscale of 57x129 = 228x516
    # But we'll infer dynamically from model output on first batch
    y25_raw = np.memmap(y25_path, dtype=np.float32, mode="r", shape=(t_total, *divmod(hw, 1)[0:1] + (1,)))
    # Simpler: load small slice to discover shape
    del y25_raw
    # Robust approach: find h25,w25 as factors closest to 4x original
    import math
    # h25 ≈ 57*4=228, w25 ≈ 129*4=516
    # Try reading y25 as (t_total, ?) and recover from flat
    flat_size = y25_path.stat().st_size // 4
    pixels_per_day = flat_size // t_total
    # Factor pixels_per_day into h*w; use known 4x factor
    h25 = int(round(math.sqrt(pixels_per_day * (57 / 129))))
    w25 = pixels_per_day // h25

    y25_mm = np.memmap(y25_path, dtype=np.float32, mode="r", shape=(t_total, h25, w25))

    elev_hr = None
    if args.elevation:
        elev_hr = _normalize_elevation(np.load(path_output_hr / "elev_lr_scaled.npy"))
        if elev_hr.shape != (h25, w25):
            raise ValueError(f"HR elevation shape {elev_hr.shape} != 25km shape {(h25, w25)}")

    # Discover output shape from first batch
    batch_t = np.asarray(y25_mm[0:1], dtype=np.float32)
    batch_s = _scale_batch_3d(batch_t, scaler_hr)
    if args.elevation:
        xp = np.stack([batch_s, np.broadcast_to(elev_hr, batch_s.shape)], axis=1)
    else:
        xp = batch_s[:, None, :, :]
    with torch.no_grad():
        out_shape = g_hr(torch.from_numpy(xp).float().to(device)).shape  # (1,1,H4,W4)
    h4_out, w4_out = out_shape[2], out_shape[3]
    print(f"[info] 4 km output shape per day: ({h4_out}, {w4_out})")

    y4_mm = np.memmap(y4_path, dtype=np.float32, mode="w+", shape=(t_total, h4_out, w4_out))
    with torch.no_grad():
        for i in tqdm(range(0, t_total, args.batch_size_hr), desc="Infer 25->4"):
            j = min(i + args.batch_size_hr, t_total)
            batch25 = np.asarray(y25_mm[i:j], dtype=np.float32)
            batch_s = _scale_batch_3d(batch25, scaler_hr)
            if args.elevation:
                x = np.stack([batch_s, np.broadcast_to(elev_hr, batch_s.shape)], axis=1)
            else:
                x = batch_s[:, None, :, :]
            pred4 = _inverse_batch_3d(
                g_hr(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()[:, 0, :, :],
                scaler_hr,
            )
            if args.var == "pr":
                np.maximum(pred4, 0, out=pred4)
            y4_mm[i:j] = pred4
    y4_mm.flush()
    del y4_mm, y25_mm
    print(f"[done] 25->4 km saved to {y4_path}")

    if args.skip_y25:
        y25_path.unlink(missing_ok=True)
        print(f"[info] removed {y25_path}")


def stage_save_nc(args, npy_dir, year_days, lat, lon, lat_dtype, lon_dtype,
                  lat_attrs, lon_attrs, out_var, default_units, output_dir, year_files):
    """CPU stage: read y_pred_4.npy and write yearly NetCDF files with time/lat/lon dims."""
    y4_path = npy_dir / "y_pred_4.npy"
    if not y4_path.exists():
        raise FileNotFoundError(f"y_pred_4.npy not found at {y4_path}. Run --mode downscale first.")

    t_total = sum(year_days)
    h4, w4 = len(lat), len(lon)

    y4_mm = np.memmap(y4_path, dtype=np.float32, mode="r", shape=(t_total, h4, w4))

    day_offset = 0
    for year_idx, year in enumerate(range(args.year_start, args.year_end)):
        days = year_days[year_idx]
        year_file = year_files[year_idx]

        if year_file.exists():
            print(f"[skip] {year_file.name} already exists")
            day_offset += days
            continue

        year_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"  writing {year} ({days} days) -> {year_file.name}")

        ds, var_out = _create_yearly_netcdf(
            year_file, year, days, lat, lon, lat_dtype, lon_dtype,
            lat_attrs, lon_attrs, out_var, default_units, args.nc_compression_level,
        )
        var_out[:, :, :] = np.asarray(y4_mm[day_offset:day_offset + days], dtype=np.float32)
        ds.close()
        day_offset += days

    del y4_mm
    print("[done] yearly NetCDF files written")
    if args.skip_y4:
        y4_path.unlink(missing_ok=True)
        print(f"[info] removed {y4_path}")


def run(args):
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    year_days = _year_lengths(args.year_start, args.year_end)

    npy_dir = base_dir / "gcm_ds" / args.downscale_version / args.var / args.gcm
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Grid info always needed for save-nc; lightweight to load.
    lat, lon, lat_dtype, lon_dtype, lat_attrs, lon_attrs, var_attrs = _read_daymet_grid(
        args.daymet_data_dir, args.var, args.year_start
    )
    out_var = args.variable_name if args.variable_name else f"{args.var}_downscaled"
    default_units = var_attrs.get("units", "unknown")

    year_files = [
        _build_output_file(
            output_dir, args.file_pattern, args.var, args.gcm,
            args.scenario, args.ensemble, args.grid, year, args.downscale_version,
        )
        for year in range(args.year_start, args.year_end)
    ]

    if args.mode in ("downscale", "all"):
        stage_downscale(args, npy_dir, year_days)

    if args.mode in ("save-nc", "all"):
        all_nc_exist = all(p.exists() for p in year_files)
        if all_nc_exist:
            print("[info] all yearly NetCDF files already exist; skipping save-nc")
        else:
            print("[stage] save-nc: writing yearly NetCDF files from y_pred_4.npy")
            stage_save_nc(
                args, npy_dir, year_days, lat, lon, lat_dtype, lon_dtype,
                lat_attrs, lon_attrs, out_var, default_units, output_dir, year_files,
            )


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
