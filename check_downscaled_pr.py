"""
Diagnostic script: inspect and plot 1 year of ACCESS-CM2 downscaled
precipitation data from SRGAN output.

Checks:
  1. y_pred_25.npy  – intermediate 25 km prediction (stored in gcm_ds/0.1/pr/ACCESS-CM2/)
  2. The first saved yearly NetCDF file (0.0416 deg / ~4 km final output)

Run with:
    /ccs/home/haoranniu/miniconda3/envs/srgan/bin/python check_downscaled_pr.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset

# ---- make sure local utils is importable ----
sys.path.insert(0, os.path.dirname(__file__))
from utils.plot_utils import plot_subplot, plot_histogram, print_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON          = "/ccs/home/haoranniu/miniconda3/envs/srgan"   # for reference
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
GCM             = "ACCESS-CM2"
SCENARIO        = "ssp585"
VAR             = "pr"
VERSION         = "0.1"

# 25 km intermediate prediction (all years concatenated)
PRED25_PATH     = f"{BASE_DIR}/gcm_ds/{VERSION}/{VAR}/{GCM}/y_pred_25.npy"

# Saved yearly NetCDF files at 0.0416 deg
NC_DIR          = f"{BASE_DIR}/gcm_ds/ssp585_yearly_nc_before_BC_0416deg/{VAR}/{GCM}"

# Coordinate reference: Daymet 0.25 deg file (matches y_pred_25 spatial extent)
DAYMET_25DEG    = (
    "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data/"
    "Daymet_ERA5_prcp_3h_198001_0p25deg.nc"
)
# Coordinate reference: Daymet trim file (matches 0.0416 deg NC output)
DAYMET_TRIM     = (
    "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data/"
    "Daymet_ERA5_prcp_3h_198001_trim.nc"
)

OUTPUT_DIR      = f"{BASE_DIR}/gcm_ds/{VERSION}/{VAR}/{GCM}/fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: load lat / lon from a Daymet NetCDF (1-D arrays, convert lon to -180..180)
# ---------------------------------------------------------------------------
def load_latlon(nc_path):
    with Dataset(nc_path) as ds:
        lat = ds.variables["lat"][:].astype(np.float32)
        lon = ds.variables["lon"][:].astype(np.float32)
    # Convert 0-360 longitude to -180..180 for cartopy
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon


# ---------------------------------------------------------------------------
# 1.  y_pred_25.npy  --------------------------------------------------------
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Checking y_pred_25.npy  (25 km intermediate prediction)")
print("="*60)

if not os.path.exists(PRED25_PATH):
    print(f"[ERROR] File not found: {PRED25_PATH}")
    sys.exit(1)

pred25 = np.load(PRED25_PATH, mmap_mode="r")         # memory-mapped, shape (T, H, W)
print(f"Full array shape : {pred25.shape}  (days x lat x lon)")
total_days = pred25.shape[0]

# Determine how many days belong to the first year.
# The historical run (v0.1) starts in 1980; 1980 is a leap year (366 days).
# Adjust FIRST_YEAR_DAYS if your run starts in a different year.
FIRST_YEAR_DAYS = 365   # change to 366 for a leap-year start
first_year = pred25[:FIRST_YEAR_DAYS]               # (FIRST_YEAR_DAYS, H, W)

print_stats(first_year, label=f"y_pred_25 — first {FIRST_YEAR_DAYS} days", var=VAR)

# ---- spatial map: annual mean ----
lat25, lon25 = load_latlon(DAYMET_25DEG)

annual_mean_25 = np.nanmean(first_year, axis=0)      # (H, W)

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

im1 = plot_subplot(axes[0], annual_mean_25, lat25, lon25,
                   title=f"{GCM}  y_pred_25  Annual Mean (mm/day) — Year 1",
                   var=VAR, stat="mean")
plt.colorbar(im1, ax=axes[0], orientation="vertical", shrink=0.75, label="mm/day")

im2 = plot_subplot(axes[1], np.nanpercentile(first_year, 95, axis=0),
                   lat25, lon25,
                   title=f"{GCM}  y_pred_25  95th Percentile (mm/day) — Year 1",
                   var=VAR, stat="95th")
plt.colorbar(im2, ax=axes[1], orientation="vertical", shrink=0.75, label="mm/day")

fig.suptitle(f"SRGAN 25 km Downscaled Precipitation — {GCM} {SCENARIO}", fontsize=13)
plt.tight_layout()
out_map25 = os.path.join(OUTPUT_DIR, "check_pred25_annual_map.png")
plt.savefig(out_map25, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nMap saved → {out_map25}")

# ---- histogram ----
fig, ax = plt.subplots(figsize=(7, 4))
plot_histogram(ax, first_year, var=VAR,
               label=f"y_pred_25 Year-1 daily values (n={FIRST_YEAR_DAYS} days)")
ax.set_title(f"{GCM} — 25 km prediction — value distribution")
plt.tight_layout()
out_hist25 = os.path.join(OUTPUT_DIR, "check_pred25_histogram.png")
plt.savefig(out_hist25, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Histogram saved → {out_hist25}")


# ---------------------------------------------------------------------------
# 2.  y_pred_4.npy  ---------------------------------------------------------
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Checking y_pred_4.npy  (4 km / 0.0416° final prediction)")
print("="*60)

PRED4_PATH = f"{BASE_DIR}/gcm_ds/{VERSION}/{VAR}/{GCM}/y_pred_4.npy"

if not os.path.exists(PRED4_PATH):
    print(f"[WARN] File not found: {PRED4_PATH}")
else:
    # y_pred_4 is a raw float32 memmap (not an npy header file)
    T4, H4, W4 = 14610, 1368, 3096  # 6× upsampled from y_pred_25
    fsize = os.path.getsize(PRED4_PATH)
    expected = T4 * H4 * W4 * 4
    print(f"File size : {fsize:,} bytes  (expected {expected:,}  match={fsize==expected})")

    raw4 = np.memmap(PRED4_PATH, dtype=np.float32, mode="r", shape=(T4, H4, W4))
    first_year4 = np.array(raw4[:FIRST_YEAR_DAYS])   # load first year into RAM

    print_stats(first_year4, label=f"y_pred_4 — first {FIRST_YEAR_DAYS} days", var=VAR)

    lat_trim, lon_trim = load_latlon(DAYMET_TRIM)
    annual_mean_4 = np.nanmean(first_year4, axis=0)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    vmax4 = max(float(np.nanpercentile(first_year4, 99)), 0.01)
    im1 = plot_subplot(axes[0], annual_mean_4, lat_trim, lon_trim,
                       title=f"{GCM}  y_pred_4  Annual Mean — Year 1",
                       var=VAR, stat="mean", vmin=0, vmax=vmax4)
    plt.colorbar(im1, ax=axes[0], orientation="vertical", shrink=0.75, label="stored units")

    im2 = plot_subplot(axes[1], np.nanpercentile(first_year4, 95, axis=0),
                       lat_trim, lon_trim,
                       title=f"{GCM}  y_pred_4  95th Pctile — Year 1",
                       var=VAR, stat="95th", vmin=0, vmax=vmax4 * 5)
    plt.colorbar(im2, ax=axes[1], orientation="vertical", shrink=0.75, label="stored units")

    fig.suptitle(
        f"SRGAN 4 km Downscaled Precipitation — {GCM} {SCENARIO}  "
        f"(mean={float(np.nanmean(first_year4)):.4f})",
        fontsize=11,
    )
    plt.tight_layout()
    out_map4 = os.path.join(OUTPUT_DIR, "check_pred4_annual_map.png")
    plt.savefig(out_map4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMap saved → {out_map4}")

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_histogram(ax, first_year4, var=VAR,
                   label=f"y_pred_4 Year-1 daily values (n={FIRST_YEAR_DAYS} days)")
    ax.set_title(f"{GCM} — 4 km prediction — value distribution")
    plt.tight_layout()
    out_hist4 = os.path.join(OUTPUT_DIR, "check_pred4_histogram.png")
    plt.savefig(out_hist4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved → {out_hist4}")


# ---------------------------------------------------------------------------
# 4.  Saved yearly NetCDF  --------------------------------------------------
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Checking saved yearly NetCDF  (0.0416 deg / 4 km final output)")
print("="*60)

nc_files = sorted(f for f in os.listdir(NC_DIR) if f.endswith(".nc")) if os.path.isdir(NC_DIR) else []
if not nc_files:
    print(f"[WARN] No .nc files found in {NC_DIR}")
else:
    first_nc = os.path.join(NC_DIR, nc_files[0])
    print(f"Loading : {first_nc}")

    with Dataset(first_nc) as ds:
        # The xarray-saved variable name may vary; grab the first non-coord variable
        data_vars = [v for v in ds.variables
                     if v not in ("lat", "lon", "time", "dim_0", "dim_1", "dim_2", "dim_3")]
        var_name = data_vars[0] if data_vars else list(ds.variables.keys())[0]
        nc_data = ds.variables[var_name][:].astype(np.float32)   # (days, lat, lon, [1])

    # Drop trailing size-1 dimension if present (artefact of np.expand_dims in converter)
    if nc_data.ndim == 4 and nc_data.shape[-1] == 1:
        nc_data = nc_data[..., 0]

    print_stats(nc_data, label=f"NetCDF '{var_name}' — {nc_files[0]}", var=VAR)

    lat_trim, lon_trim = load_latlon(DAYMET_TRIM)

    # ---- spatial map ----
    annual_mean_nc = np.nanmean(nc_data, axis=0)   # (H, W)

    # Determine actual value range so we do not clip small values
    vmax_nc = float(np.nanpercentile(nc_data, 99))
    vmin_nc = 0.0

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    im1 = plot_subplot(axes[0], annual_mean_nc, lat_trim, lon_trim,
                       title=f"{GCM} NetCDF Annual Mean — {nc_files[0][:4]} (mm/day?)",
                       var=VAR, stat="mean", vmin=vmin_nc, vmax=vmax_nc)
    plt.colorbar(im1, ax=axes[0], orientation="vertical", shrink=0.75, label="stored units")

    im2 = plot_subplot(axes[1], np.nanpercentile(nc_data, 95, axis=0),
                       lat_trim, lon_trim,
                       title=f"{GCM} NetCDF 95th Pctile — {nc_files[0][:4]}",
                       var=VAR, stat="95th", vmin=vmin_nc, vmax=vmax_nc * 5)
    plt.colorbar(im2, ax=axes[1], orientation="vertical", shrink=0.75, label="stored units")

    fig.suptitle(
        f"SRGAN 0.0416° Downscaled Precipitation — {GCM} {SCENARIO}\n"
        f"(mean={float(np.nanmean(nc_data)):.4f}, if small → possible unit bug)",
        fontsize=11,
    )
    plt.tight_layout()
    out_map_nc = os.path.join(OUTPUT_DIR, f"check_nc_annual_map_{nc_files[0][:4]}.png")
    plt.savefig(out_map_nc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMap saved → {out_map_nc}")

    # ---- histogram ----
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_histogram(ax, nc_data, var=VAR,
                   label=f"NetCDF {nc_files[0][:4]} daily values")
    ax.set_title(f"{GCM} — 0.0416° NetCDF — value distribution")
    plt.tight_layout()
    out_hist_nc = os.path.join(OUTPUT_DIR, f"check_nc_histogram_{nc_files[0][:4]}.png")
    plt.savefig(out_hist_nc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved → {out_hist_nc}")

    # ---- comparison summary ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    mean25 = float(np.nanmean(first_year))
    mean4  = float(np.nanmean(first_year4)) if os.path.exists(PRED4_PATH) else float('nan')
    meannc = float(np.nanmean(nc_data))
    print(f"  y_pred_25 (25 km) year-1 mean : {mean25:.4f} mm/day")
    print(f"  y_pred_4  (4 km)  year-1 mean : {mean4:.4f}")
    print(f"  NetCDF (4 km)     year-1 mean : {meannc:.4f}")
    ratio = mean25 / (mean4 + 1e-12)
    print(f"  Ratio pred25 / pred4          : {ratio:.1f}")
    if ratio > 10:
        print("  [WARNING] y_pred_4 values are much smaller than y_pred_25.")
        print("            Root cause: downscale_25_4() used self.loaded_scaler")
        print("            (LR scaler) instead of self.loaded_scaler_hr (HR scaler)")
        print("            for the inverse transform — values stay in normalised [0,1]")
        print("            space instead of being converted back to mm/day.")
        print("            Fix applied in gcm_downscaling_prcp.py; re-run downscaling.")

print("\nDone.")
